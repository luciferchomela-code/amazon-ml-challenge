"""
step2_normalization.py
======================
Business Entity Resolution Pipeline — Step 2: Text Normalization & Standardization

Scale  : 1M+ records per source
Outputs:
    Returns the same dictionary of DataFrames from Step 1, but enriched with:
    - clean_name      : fully cleaned, acronym-squeezed, abbreviation-expanded
    - clean_address   : fully cleaned, abbreviation-expanded
    - sorted_name     : token-sorted version of clean_name (word-order invariant)
    - phonetic_name   : Double Metaphone hash of the first token of sorted_name
    - postal_code     : Extracted US/India/France postal codes

CHANGELOG (patched):
    FIX #1: "&" is now converted to " and " BEFORE punctuation stripping, so
            "Smith & Sons" and "Smith and Sons" normalize identically instead
            of silently diverging ("smith sons" vs "smith and sons").
    FIX #2: Phonetic hashing now runs on the first token of `sorted_name`
            instead of `clean_name`, so word-order-transposed names
            ("Taj Hotel" vs "Hotel Taj") produce the same phonetic code
            instead of hashing whichever word happened to come first.

Dependencies:
    pip install polars>=0.20.0 flashtext>=2.7 jellyfish>=1.0.3

Usage:
    from step2_normalization import run_normalization
    norm_data = run_normalization(data_dict_from_step1)
"""

import os
import re
import sys
import time
import multiprocessing as mp
from typing import List, Tuple

import polars as pl
from flashtext import KeywordProcessor
import jellyfish

# ─────────────────────────────────────────────────────────────
# COMPILED REGEXES
# ─────────────────────────────────────────────────────────────

# Postal codes: 5 digits (US/France) or 5 digits + 4 (US Zip+4) or 6 digits (India)
POSTAL_RE = r"\b(\d{5}(?:-\d{4})?|\d{6})\b"

# Acronyms: finds spaces between single lowercase letters.
# e.g., "i b m" -> "ibm"
# (Used via Python's `re` module because Rust/Polars regex doesn't support lookarounds)
ACRONYM_RE = re.compile(r"(?<=\b[a-z])\s+(?=[a-z]\b)")


# ─────────────────────────────────────────────────────────────
# FLASHTEXT SETUP
# ─────────────────────────────────────────────────────────────

def setup_keyword_processors() -> Tuple[KeywordProcessor, KeywordProcessor]:
    """
    Initializes Aho-Corasick automata for ultra-fast abbreviation expansion.
    Complexity is O(N) based on the input text length, completely independent
    of the dictionary size.

    NOTE: The "and": ["&"] mapping that used to live here has been REMOVED.
    It never fired because the global clean pass already stripped "&" before
    FlashText ran. The fix now lives in `_clean_col` (see FIX #1 below),
    which converts "&" -> " and " BEFORE punctuation stripping.
    """
    name_kp = KeywordProcessor()
    name_kp.add_keywords_from_dict({
        "corporation": ["corp", "corpn", "crp"],
        "incorporated": ["inc", "incorp", "incorporat"],
        "private": ["pvt", "prvt", "pte"],
        "limited": ["ltd", "lmtd"],
        "company": ["co", "coy"],
        "industries": ["inds", "ind", "indust"],
        "enterprises": ["ent", "entp", "enterp"],
        "services": ["svc", "svcs", "serv", "service"],
        "international": ["intl", "int", "internat"],
        "technologies": ["tech", "techno"],
    })

    addr_kp = KeywordProcessor()
    addr_kp.add_keywords_from_dict({
        "street": ["st", "str"],
        "road": ["rd"],
        "avenue": ["ave", "aven"],
        "boulevard": ["blvd"],
        "highway": ["hwy"],
        "north": ["n", "no"],
        "south": ["s", "so"],
        "east": ["e", "ea"],
        "west": ["w", "we"],
        "apartment": ["apt", "apart"],
        "building": ["bldg", "build"],
        "floor": ["fl", "flr"],
        "room": ["rm"],
        "suite": ["ste"],
        "near": ["nr"],
        "opposite": ["opp"],
    })

    return name_kp, addr_kp


# ─────────────────────────────────────────────────────────────
# PYTHON-LEVEL TEXT PROCESSING (Acronyms & FlashText)
# ─────────────────────────────────────────────────────────────

def process_text_python(texts: List[str], kp: KeywordProcessor, squeeze_acronyms: bool = False) -> List[str]:
    """
    Apply regex lookarounds (acronyms) and FlashText replacements.
    This runs in pure Python list comprehensions, which are highly optimized
    and actually out-perform Polars `.map_elements` for these specific tasks.
    """
    results = []
    for t in texts:
        if not t:
            results.append("")
            continue

        if squeeze_acronyms:
            # "i b m" -> "ibm"
            t = ACRONYM_RE.sub("", t)

        # "starbucks corp" -> "starbucks corporation"
        t = kp.replace_keywords(t)
        results.append(t)

    return results


# ─────────────────────────────────────────────────────────────
# PARALLEL PHONETIC HASHING
# ─────────────────────────────────────────────────────────────

def _phonetic_batch(texts: List[str]) -> List[str]:
    """
    Worker function for multiprocessing.

    FIX #2: This now expects `sorted_name` (token-sorted) as input, not raw
    `clean_name`. Hashing the first token of the ALPHABETICALLY sorted name
    makes the phonetic code invariant to word-order transpositions
    ("Taj Hotel" vs "Hotel Taj" -> both sort to "hotel taj" -> same first
    token "hotel" -> same metaphone code either way).
    """
    res = []
    for t in texts:
        if not t:
            res.append("")
        else:
            first_word = t.split()[0]
            res.append(jellyfish.metaphone(first_word))
    return res


def parallel_phonetic(
    series: pl.Series,
    n_workers: int = None,
    chunk_size: int = 100_000,
) -> pl.Series:
    """
    Computes Double Metaphone hashes using all CPU cores.
    Bypasses the GIL and prevents the 45-minute `.apply()` lockup.
    """
    if n_workers is None:
        n_workers = max(1, min(8, mp.cpu_count() - 1))

    texts = series.to_list()
    if not texts:
        return pl.Series(name="phonetic_name", values=[])

    # Avoid macOS process-spawn/resource failures on multi-million-row tables.
    if len(texts) > 500_000:
        return pl.Series(name="phonetic_name", values=_phonetic_batch(texts))

    flat_results = []
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start:start + chunk_size]
        batch_size = max(1, len(chunk) // n_workers)
        batches = [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
        with mp.Pool(min(n_workers, len(batches))) as pool:
            results = pool.map(_phonetic_batch, batches)
        flat_results.extend(code for batch in results for code in batch)

    return pl.Series(name="phonetic_name", values=flat_results)


# ─────────────────────────────────────────────────────────────
# COLLISION TEST (OVER-NORMALIZATION GUARD)
# ─────────────────────────────────────────────────────────────

def collision_test(df_s1: pl.DataFrame, col_name: str, label: str) -> None:
    """
    S1 is a deduplicated reference source.
    If two different S1 entity_ids collapse to the EXACT same normalized string,
    it signals that our normalization might be destroying too much information.
    """
    # Ignore empty strings for this check
    df_valid = df_s1.filter(pl.col(col_name) != "")

    collisions = (
        df_valid.group_by(col_name)
                .agg(pl.count().alias("n"))
                .filter(pl.col("n") > 1)
    )

    if collisions.height > 0:
        print(f"  [WARN]  [{label}] {collisions.height:,} S1 collisions on '{col_name}' — check for over-normalization")
    else:
        print(f"  [OK]    [{label}] No S1 collisions on '{col_name}'")


# ─────────────────────────────────────────────────────────────
# CORE DATAFRAME NORMALIZATION
# ─────────────────────────────────────────────────────────────

def normalize_dataframe(
    df: pl.DataFrame,
    label: str,
    name_kp: KeywordProcessor,
    addr_kp: KeywordProcessor
) -> pl.DataFrame:
    """
    Runs the full optimized pipeline on a single Polars DataFrame.
    """
    start_time = time.time()
    print(f"\n  [START] [{label}] Normalizing {df.height:,} rows...")

    # 1. Postal Code Extraction (MUST happen before global clean destroys punctuation)
    if "postal_code" not in df.columns:
        df = df.with_columns(
            pl.col("business_address").str.extract(POSTAL_RE, 1).alias("postal_code")
        )

    # 2. Global Clean (Rust/SIMD Regex)
    #
    # FIX #1: "&" -> " and " is applied BEFORE the alnum-strip regex.
    # Previously "&" was destroyed by `[^a-z0-9\s]` before FlashText ever
    # saw it, so "Smith & Sons" ("smith sons") and "Smith and Sons"
    # ("smith and sons") diverged instead of matching, even though the
    # problem statement explicitly calls out "& vs. and" as expected noise.
    def _clean_col(col_name: str, new_col_name: str) -> pl.Expr:
        return (
            pl.col(col_name)
            .str.to_lowercase()
            .str.replace_all(r"&", " and ", literal=True)          # FIX #1
            .str.replace_all(r"[^a-z0-9\s]", " ", literal=False)
            .str.replace_all(r"\s+", " ", literal=False)
            .str.strip_chars()
            .alias(new_col_name)
        )

    df = df.with_columns([
        _clean_col("business_name", "clean_name"),
        _clean_col("business_address", "clean_address"),
    ])

    # 3. Python-Level Processing (Acronyms & FlashText)
    names_raw = df["clean_name"].to_list()
    addrs_raw = df["clean_address"].to_list()

    names_proc = process_text_python(names_raw, name_kp, squeeze_acronyms=True)
    addrs_proc = process_text_python(addrs_raw, addr_kp, squeeze_acronyms=False)

    df = df.with_columns([
        pl.Series("clean_name", names_proc),
        pl.Series("clean_address", addrs_proc),
    ])

    # 4. Token Sorting (Word-order invariance)
    # "atm sbi near" -> "atm near sbi"
    df = df.with_columns([
        pl.col("clean_name")
          .str.split(" ")
          .list.sort()
          .list.join(" ")
          .alias("sorted_name")
    ])

    # 5. Phonetic Hashing (Multiprocessing)
    # FIX #2: hash the SORTED name, not clean_name, for order-invariance.
    phonetic_series = parallel_phonetic(df["sorted_name"])
    df = df.with_columns(phonetic_series)

    elapsed = time.time() - start_time
    print(f"  [DONE]  [{label}] Finished in {elapsed:.1f}s")

    return df


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────

def run_normalization(data_dict: dict, cache_dir: str = "output/cache", force_reload: bool = False) -> dict:
    """
    Normalize all loaded DataFrames.

    Parameters
    ----------
    data_dict    : Dictionary of DataFrames output by Step 1.
    cache_dir    : Where to save the normalized Parquet caches.
    force_reload : If True, ignore existing normalized caches and re-process.

    Returns
    -------
    Enriched dictionary of DataFrames.
    """
    print("\n" + "=" * 60)
    print("STEP 2: TEXT NORMALIZATION & STANDARDIZATION")
    print("=" * 60)

    os.makedirs(cache_dir, exist_ok=True)
    name_kp, addr_kp = setup_keyword_processors()

    out_dict = data_dict.copy()

    # We don't normalize the ground truth
    out_dict["gt"] = data_dict["gt"]
    out_dict["new_test_countries"] = data_dict.get("new_test_countries", set())

    # Target keys to process
    keys_to_process = ["s1_train", "s23_train", "s1_test", "s23_test"]

    for key in keys_to_process:
        if key not in data_dict:
            continue

        cache_path = os.path.join(cache_dir, f"{key}_normalized.parquet")

        # Fast path
        if os.path.exists(cache_path) and not force_reload:
            print(f"  [CACHE] Loading {key} (normalized) from Parquet...")
            df_norm = pl.read_parquet(cache_path)
        else:
            # Slow path
            df_raw = data_dict[key]
            df_norm = normalize_dataframe(df_raw, key.upper(), name_kp, addr_kp)
            df_norm.write_parquet(cache_path)

        out_dict[key] = df_norm
        # Release the raw frame before normalizing the next multi-million-row table.
        data_dict[key] = df_norm

        # Run S1 collision guard checks
        if "s1" in key:
            collision_test(df_norm, "clean_name", key.upper())
            collision_test(df_norm, "sorted_name", key.upper())

    print("\n" + "=" * 60)
    print("STEP 2 COMPLETE")
    print("=" * 60)
    return out_dict


# ─────────────────────────────────────────────────────────────
# STANDALONE EXECUTION
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from step1_ingestion import run_ingestion

    parser = argparse.ArgumentParser(description="Step 2: Text Normalization")
    parser.add_argument("--train-dir",    default="dataset/train", help="Training data directory")
    parser.add_argument("--test-dir",     default="dataset/test",  help="Test data directory")
    parser.add_argument("--force-reload", action="store_true",     help="Ignore caches and re-process")
    args = parser.parse_args()

    # Step 1
    step1_data = run_ingestion(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        force_reload=args.force_reload
    )

    # Step 2
    step2_data = run_normalization(
        step1_data,
        force_reload=args.force_reload
    )
