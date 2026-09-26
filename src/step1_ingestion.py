"""
step1_ingestion.py
==================
Business Entity Resolution Pipeline — Step 1: Data Ingestion & Schema Alignment

Scale  : 1M+ records per source
Outputs:
    df_s1_train  : Polars DataFrame — Source 1 training records
    df_s23_train : Polars DataFrame — Source 2+3 combined training records
    df_s1_test   : Polars DataFrame — Source 1 test records
    df_s23_test  : Polars DataFrame — Source 2+3 combined test records
    df_gt        : Polars DataFrame — Ground truth (train only)

Dependencies:
    pip install polars>=0.20.0 pyarrow>=14.0.0

Usage:
    from step1_ingestion import run_ingestion
    data = run_ingestion(train_dir="dataset/train", test_dir="dataset/test")
    df_s1_train  = data["s1_train"]
    df_s23_train = data["s23_train"]
    df_s1_test   = data["s1_test"]
    df_s23_test  = data["s23_test"]
    df_gt        = data["gt"]
"""

import os
import sys
import polars as pl
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

REQUIRED_COLS     = {"entity_id", "business_name", "business_address", "country"}
TEXT_COLS         = ["business_name", "business_address", "country"]
S1_PREFIX         = "S1-"
S23_PREFIXES      = ("S2-", "S3-")

# These are treated as true null — empty string "" is NOT null here
# because "no address" is valid information and should stay as ""
NULL_MARKERS      = ["NA", "N/A", "null", "NULL", "none", "NONE", "nan", "NaN"]


# ─────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────

def _get_cache_path(raw_path: str, cache_dir: str) -> str:
    """Derive the Parquet cache path from the raw TSV path."""
    stem = Path(raw_path).stem
    return os.path.join(cache_dir, f"{stem}.parquet")


def _load_raw_tsv(path: str) -> pl.DataFrame:
    """
    Low-level TSV reader with all safety flags set.

    Key flags:
    - separator="\\t"          : explicit tab — never guess
    - infer_schema_length=0   : all columns as Utf8 string — prevents silent
                                 float/int conversion of entity_ids or codes
    - quote_char=None          : business names/addresses contain quotation marks;
                                 disabling quote processing prevents column shifts
    - encoding="utf8-lossy"   : handles BOM (\\ufeff) and mixed-encoding files
                                 silently — critical for data from multiple countries
    - truncate_ragged_lines    : skips malformed rows instead of crashing
    - null_values=NULL_MARKERS : only canonical null markers, NOT empty string
    """
    if not os.path.exists(path):
        sys.exit(f"[FATAL] File not found: {path}")

    return pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=0,
        quote_char=None,
        encoding="utf8-lossy",
        truncate_ragged_lines=True,
        null_values=NULL_MARKERS,
        ignore_errors=True,
    )


# ─────────────────────────────────────────────────────────────
# VALIDATION FUNCTIONS
# ─────────────────────────────────────────────────────────────

def validate_schema(df: pl.DataFrame, label: str) -> pl.DataFrame:
    """
    Verify required columns exist and drop any unexpected extra columns.
    Exits immediately on missing required columns — never silently continue.
    """
    actual  = set(df.columns)
    missing = REQUIRED_COLS - actual
    extra   = actual - REQUIRED_COLS

    if missing:
        sys.exit(f"[FATAL] [{label}] Missing required columns: {missing}")

    if extra:
        print(f"  [WARN]  [{label}] Dropping unexpected columns: {extra}")
        df = df.select(sorted(REQUIRED_COLS))

    return df


def validate_entity_id_prefixes(df: pl.DataFrame, label: str, valid_prefixes: tuple) -> None:
    """
    Ensure every entity_id starts with one of the expected prefixes.
    Catches file loading errors (wrong file, column shift, etc.).
    Vectorized — never uses .apply() or Python loops.
    """
    # Build mask dynamically — works for any open set of prefixes
    mask = pl.lit(False)
    for prefix in valid_prefixes:
        mask = mask | pl.col("entity_id").str.starts_with(prefix)

    bad = df.filter(~mask)
    if not bad.is_empty():
        print(f"  [FATAL] [{label}] {bad.height:,} entity_ids have wrong prefix! Examples:")
        print(bad.head(5))
        sys.exit(1)


def check_and_remove_duplicates(df: pl.DataFrame, label: str) -> pl.DataFrame:
    """
    Detect and remove duplicate entity_ids within a source file.
    Per problem statement, Source 1 is deduplicated — any duplicate signals a data error.
    S2/S3 duplicates are also flagged (they should not exist per problem spec).
    Keeps the first occurrence when deduplication is needed.
    """
    total  = df.height
    unique = df["entity_id"].n_unique()
    n_dups = total - unique

    if n_dups > 0:
        duped_keys = (
            df.group_by("entity_id")
              .agg(pl.count().alias("n"))
              .filter(pl.col("n") > 1)
        )
        print(f"  [WARN]  [{label}] {n_dups:,} duplicate entity_ids — keeping first. Sample:")
        print(duped_keys.head(5))
        df = df.unique(subset=["entity_id"], keep="first", maintain_order=True)
    else:
        print(f"  [OK]    [{label}] No duplicate entity_ids ({total:,} unique)")

    return df


def fill_text_nulls(df: pl.DataFrame, label: str) -> pl.DataFrame:
    """
    Fill null values in text columns with empty string "".
    - Does NOT fill with 'unknown' or any synthetic token — that creates fake
      signals that degrade feature quality downstream.
    - Logs how many nulls were filled per column for transparency.
    """
    for col in TEXT_COLS:
        if col not in df.columns:
            continue
        n_null = df[col].is_null().sum()
        if n_null > 0:
            print(f"  [NULL]  [{label}][{col}] {n_null:,} nulls filled with ''")
        df = df.with_columns(pl.col(col).fill_null(""))
    return df


# ─────────────────────────────────────────────────────────────
# ENRICHMENT
# ─────────────────────────────────────────────────────────────

def add_source_tag(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add 'source_tag' column ("S1", "S2", "S3") derived from entity_id prefix.
    Fully vectorized via Polars str.slice — no Python loop.

    "S1-00001" → "S1"
    "S2-99999" → "S2"
    """
    return df.with_columns(
        pl.col("entity_id").str.slice(0, 2).alias("source_tag")
    )


# ─────────────────────────────────────────────────────────────
# QUALITY REPORT
# ─────────────────────────────────────────────────────────────

def print_quality_report(df: pl.DataFrame, label: str) -> None:
    """Print a concise, actionable data quality summary."""
    countries    = sorted(df["country"].drop_nulls().unique().to_list())
    empty_name   = (df["business_name"]    == "").sum()
    empty_addr   = (df["business_address"] == "").sum()
    empty_ctry   = (df["country"]          == "").sum()

    print(f"\n  ── Quality Report [{label}] ──────────────────────────")
    print(f"  Rows              : {df.height:,}")
    print(f"  Countries (unique): {countries}")
    print(f"  Empty name        : {empty_name:,}  ({empty_name/df.height*100:.2f}%)")
    print(f"  Empty address     : {empty_addr:,}  ({empty_addr/df.height*100:.2f}%)")
    print(f"  Empty country     : {empty_ctry:,}  ({empty_ctry/df.height*100:.2f}%)")
    print(f"  Sample row:\n{df.head(1)}")
    print(f"  ─────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────
# FULL LOAD PIPELINE FOR ONE SOURCE FILE
# ─────────────────────────────────────────────────────────────

def load_source_file(
    path: str,
    label: str,
    valid_prefixes: tuple,
    cache_dir: str,
    force_reload: bool = False,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Complete load, validate, and enrich pipeline for one source file.

    Steps executed:
    1. Load from Parquet cache (fast path) or raw TSV (slow path)
    2. Schema validation — crash early on wrong columns
    3. Entity ID prefix validation — catch file mix-ups
    4. Duplicate detection and removal
    5. Null filling in text columns
    6. Source tag enrichment
    7. Quality report
    8. Write to Parquet cache for future fast loads

    Parameters
    ----------
    path           : Path to the .tsv source file
    label          : Human-readable name for logging (e.g., "S1_TRAIN")
    valid_prefixes : Tuple of allowed entity_id prefixes, e.g., ("S1-",)
    cache_dir      : Directory to cache Parquet files
    force_reload   : If True, ignore cache and re-read from raw TSV
    verbose        : If True, print quality report
    """
    cache_path = _get_cache_path(path, cache_dir)

    # ── Fast path: reload from Parquet cache ──────────────────
    if os.path.exists(cache_path) and not force_reload:
        df = pl.read_parquet(cache_path)
        if verbose:
            print(f"  [CACHE] [{label}] {df.height:,} rows loaded from Parquet cache")
        return df

    # ── Slow path: read from raw TSV ──────────────────────────
    print(f"\n  [LOAD]  [{label}] Reading: {path}")
    df = _load_raw_tsv(path)
    print(f"  [LOAD]  [{label}] Raw rows read: {df.height:,} | Columns: {df.columns}")

    # ── Validate and clean ─────────────────────────────────────
    df = validate_schema(df, label)
    validate_entity_id_prefixes(df, label, valid_prefixes)
    df = check_and_remove_duplicates(df, label)
    df = fill_text_nulls(df, label)
    df = add_source_tag(df)

    # ── Report ─────────────────────────────────────────────────
    if verbose:
        print_quality_report(df, label)

    # ── Cache to Parquet for fast future reloads ───────────────
    os.makedirs(cache_dir, exist_ok=True)
    df.write_parquet(cache_path)
    print(f"  [CACHE] [{label}] Saved to {cache_path}")

    return df


# ─────────────────────────────────────────────────────────────
# GROUND TRUTH LOADER
# ─────────────────────────────────────────────────────────────

def load_ground_truth(path: str, cache_dir: str, force_reload: bool = False) -> pl.DataFrame:
    """
    Load and validate the ground truth TSV.
    - matched_entity_ids is left as a string (comma-separated) — not exploded here
    - Singletons (entities with no matches) have matched_entity_ids = ""
    """
    cache_path = _get_cache_path(path, cache_dir)

    if os.path.exists(cache_path) and not force_reload:
        df = pl.read_parquet(cache_path)
        print(f"  [CACHE] [GT] {df.height:,} rows loaded from Parquet cache")
        return df

    print(f"\n  [LOAD]  [GT] Reading: {path}")
    df = pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=0,
        null_values=NULL_MARKERS,
        encoding="utf8-lossy",
        truncate_ragged_lines=True,
        ignore_errors=True,
    )

    required_gt = {"source1_entity_id", "matched_entity_ids"}
    if not required_gt.issubset(set(df.columns)):
        sys.exit(f"[FATAL] Ground truth missing columns. Got: {df.columns}")

    df = df.with_columns(pl.col("matched_entity_ids").fill_null(""))

    singletons = (df["matched_entity_ids"] == "").sum()
    print(f"  [GT]    Total S1 entities : {df.height:,}")
    print(f"  [GT]    Singletons (no match): {singletons:,}  ({singletons/df.height*100:.1f}%)")

    os.makedirs(cache_dir, exist_ok=True)
    df.write_parquet(cache_path)
    return df


# ─────────────────────────────────────────────────────────────
# COUNTRY CONSISTENCY CHECK (OPEN-SET GUARD)
# ─────────────────────────────────────────────────────────────

def check_country_open_set(
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
    label: str = "",
) -> set:
    """
    Compare train vs test country labels.
    Prints any new countries in the test set so you are not surprised by France etc.
    Returns the set of new test-only countries.
    """
    train_countries = set(df_train["country"].drop_nulls().unique().to_list())
    test_countries  = set(df_test["country"].drop_nulls().unique().to_list())
    new_in_test     = test_countries - train_countries

    print(f"\n  [COUNTRY CHECK] {label}")
    print(f"  Train countries : {sorted(train_countries)}")
    print(f"  Test  countries : {sorted(test_countries)}")
    if new_in_test:
        print(f"  NEW in test     : {sorted(new_in_test)}  ← Open-set — pipeline handles these automatically")
    else:
        print(f"  No new countries in test set")

    return new_in_test


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def run_ingestion(
    train_dir: str = "dataset/train",
    test_dir: str  = "dataset/test",
    cache_dir: str = "output/cache",
    force_reload: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run the complete Step 1 ingestion pipeline.

    Parameters
    ----------
    train_dir    : Path to training data directory
    test_dir     : Path to test data directory
    cache_dir    : Directory for Parquet cache files
    force_reload : If True, ignore Parquet cache and re-read all TSVs
    verbose      : If True, print quality reports

    Returns
    -------
    dict with keys:
        "s1_train"  : df_s1_train
        "s23_train" : df_s23_train  (S2 + S3 combined)
        "s1_test"   : df_s1_test
        "s23_test"  : df_s23_test   (S2 + S3 combined)
        "gt"        : df_gt
        "new_test_countries" : set of country labels new in test set
    """
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)

    print("\n" + "=" * 60)
    print("STEP 1: DATA INGESTION & SCHEMA ALIGNMENT")
    print(f"  train_dir    : {train_dir}")
    print(f"  test_dir     : {test_dir}")
    print(f"  cache_dir    : {cache_dir}")
    print(f"  force_reload : {force_reload}")
    print("=" * 60)

    # ── Load training sources ──────────────────────────────────
    df_s1_train = load_source_file(
        path=f"{train_dir}/train_source1.tsv",
        label="S1_TRAIN",
        valid_prefixes=(S1_PREFIX,),
        cache_dir=cache_dir,
        force_reload=force_reload,
        verbose=verbose,
    )
    df_s2_train = load_source_file(
        path=f"{train_dir}/train_source2.tsv",
        label="S2_TRAIN",
        valid_prefixes=S23_PREFIXES,
        cache_dir=cache_dir,
        force_reload=force_reload,
        verbose=verbose,
    )
    df_s3_train = load_source_file(
        path=f"{train_dir}/train_source3.tsv",
        label="S3_TRAIN",
        valid_prefixes=S23_PREFIXES,
        cache_dir=cache_dir,
        force_reload=force_reload,
        verbose=verbose,
    )

    # ── Load test sources ──────────────────────────────────────
    df_s1_test = load_source_file(
        path=f"{test_dir}/test_source1.tsv",
        label="S1_TEST",
        valid_prefixes=(S1_PREFIX,),
        cache_dir=cache_dir,
        force_reload=force_reload,
        verbose=verbose,
    )
    df_s2_test = load_source_file(
        path=f"{test_dir}/test_source2.tsv",
        label="S2_TEST",
        valid_prefixes=S23_PREFIXES,
        cache_dir=cache_dir,
        force_reload=force_reload,
        verbose=verbose,
    )
    df_s3_test = load_source_file(
        path=f"{test_dir}/test_source3.tsv",
        label="S3_TEST",
        valid_prefixes=S23_PREFIXES,
        cache_dir=cache_dir,
        force_reload=force_reload,
        verbose=verbose,
    )

    # ── Load ground truth ──────────────────────────────────────
    df_gt = load_ground_truth(
        path=f"{train_dir}/train_ground_truth.tsv",
        cache_dir=cache_dir,
        force_reload=force_reload,
    )

    # ── Combine S2 + S3 into unified candidate pools ───────────
    df_s23_train = pl.concat([df_s2_train, df_s3_train], how="vertical_relaxed")
    df_s23_test  = pl.concat([df_s2_test,  df_s3_test],  how="vertical_relaxed")

    # ── Safety guard: no S1- IDs in S23 pool ──────────────────
    s1_in_s23_train = df_s23_train.filter(pl.col("entity_id").str.starts_with("S1-"))
    s1_in_s23_test  = df_s23_test.filter(pl.col("entity_id").str.starts_with("S1-"))
    if not s1_in_s23_train.is_empty() or not s1_in_s23_test.is_empty():
        sys.exit("[FATAL] S1- entity_ids found in S23 candidate pool. Check file loading.")

    # ── Open-set country check ─────────────────────────────────
    new_countries = check_country_open_set(df_s1_train, df_s1_test, label="S1 Train vs Test")

    # ── Final summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE — FINAL SIZES")
    print("=" * 60)
    print(f"  df_s1_train   : {df_s1_train.height:,} rows")
    print(f"  df_s23_train  : {df_s23_train.height:,} rows  "
          f"(S2: {df_s2_train.height:,} + S3: {df_s3_train.height:,})")
    print(f"  df_s1_test    : {df_s1_test.height:,} rows")
    print(f"  df_s23_test   : {df_s23_test.height:,} rows  "
          f"(S2: {df_s2_test.height:,} + S3: {df_s3_test.height:,})")
    print(f"  df_gt         : {df_gt.height:,} rows")
    print(f"  Columns       : {df_s1_train.columns}")
    print("=" * 60)

    return {
        "s1_train":           df_s1_train,
        "s23_train":          df_s23_train,
        "s1_test":            df_s1_test,
        "s23_test":           df_s23_test,
        "gt":                 df_gt,
        "new_test_countries": new_countries,
    }


# ─────────────────────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 1: Data Ingestion & Schema Alignment")
    parser.add_argument("--train-dir",    default="dataset/train", help="Training data directory")
    parser.add_argument("--test-dir",     default="dataset/test",  help="Test data directory")
    parser.add_argument("--cache-dir",    default="output/cache",  help="Parquet cache directory")
    parser.add_argument("--force-reload", action="store_true",     help="Ignore cache and re-read TSVs")
    parser.add_argument("--quiet",        action="store_true",     help="Suppress quality reports")
    args = parser.parse_args()

    result = run_ingestion(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        cache_dir=args.cache_dir,
        force_reload=args.force_reload,
        verbose=not args.quiet,
    )
