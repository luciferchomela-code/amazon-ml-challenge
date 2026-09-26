"""
===============================================================================
Business Entity Resolution Pipeline — Step 3: Candidate Generation (Blocking)
===============================================================================

Scale  : 1M+ records per source

Outputs:
    candidate_pairs.tsv (in output folder) — strictly formatted for submission
    candidate_pairs_train.parquet (cached for Step 4)
    candidate_pairs_test.parquet (cached for Step 4)

Strategy:
    "Multi-Pass Mega-Block Pruned Joins"

    Instead of doing 3M expensive FAISS neural embeddings on CPU (which takes hours),
    we use 6 clever, fuzzy Polars joins. Polars executes these in C++ in milliseconds.

    To prevent RAM crashes (Cartesian explosion), we aggressively prune "mega-blocks"
    (e.g., stopping generic matches like "McDonalds" from matching 10,000 times).

Dependencies:
    pip install polars>=0.20.0

Usage:
    from step3_blocking import run_blocking
    data_with_candidates = run_blocking(norm_data)
"""

import os
import sys
import time
import shutil
import tempfile
import polars as pl


# ─────────────────────────────────────────────────────────────
# BLOCKING STRATEGIES (MULTI-PASS)
# ─────────────────────────────────────────────────────────────

# The 'expr' is computed on both S1 and S23.
# They are strictly partitioned by 'country' automatically.

BLOCKERS = [
    {
        "name": "Exact_Name",
        "expr": pl.col("clean_name"),
        "max_size": 1000  # Cap block size to prevent OOM
    },

    {
        "name": "Sorted_Name",
        "expr": pl.col("sorted_name"),
        "max_size": 1000
    },

    {
        "name": "Phonetic_Name",
        "expr": pl.col("phonetic_name"),
        "max_size": 2500  # Phonetic hash is fuzzy, allows slightly larger blocks
    },

    {
        # First word of name + Postal Code (e.g. "Starbucks_10001")
        "name": "Name1st_Postal",
        "expr": pl.when(pl.col("postal_code") != "").then(
                    pl.col("clean_name").str.split(" ").list.first()
                    + pl.lit("_")
                    + pl.col("postal_code")
                ).otherwise(pl.lit("")),
        "max_size": 5000
    },

    {
        # First 10 chars of name + First 10 chars of address
        "name": "Name10_Addr10",
        "expr": pl.when(pl.col("clean_address") != "").then(
                    pl.col("clean_name").str.slice(0, 10)
                    + pl.lit("_")
                    + pl.col("clean_address").str.slice(0, 10)
                ).otherwise(pl.lit("")),
        "max_size": 5000
    },

    {
        # All numbers extracted from address + First word of name
        "name": "AddrNum_Name1st",
        "expr": pl.when(
                    pl.col("clean_address").str.replace_all(
                        r"[^0-9]", "", literal=False
                    ) != ""
                ).then(
                    pl.col("clean_address").str.replace_all(
                        r"[^0-9]", "", literal=False
                    )
                    + pl.lit("_")
                    + pl.col("clean_name").str.split(" ").list.first()
                ).otherwise(pl.lit("")),
        "max_size": 5000
    },

    # ─────────────────────────────────────────────────────────
    # NEW BLOCKER: Postal Code Only
    # ─────────────────────────────────────────────────────────
    {
        "name": "Postal_Only",
        "expr": pl.col("postal_code"),
        "max_size": 3000
    }
]


# ─────────────────────────────────────────────────────────────
# CANDIDATE GENERATION (PER BLOCKER)
# ─────────────────────────────────────────────────────────────

def generate_candidates(
    df_s1: pl.DataFrame,
    df_s23: pl.DataFrame,
    block_expr: pl.Expr,
    name: str,
    max_block_size: int,
    verbose: bool = True
) -> pl.DataFrame:
    """
    Executes a single blocking pass.

    Automatically prevents Cartesian explosion by pruning mega-blocks.
    """

    # 1. Apply blocking expression
    s1 = df_s1.with_columns(block_expr.alias("block_key")).select(
        "entity_id", "country", "block_key"
    )
    s23 = df_s23.with_columns(block_expr.alias("block_key")).select(
        "entity_id", "country", "block_key"
    )

    # 2. Filter out null/empty keys
    s1 = s1.filter(
        (pl.col("block_key").is_not_null())
        & (pl.col("block_key") != "")
    )

    s23 = s23.filter(
        (pl.col("block_key").is_not_null())
        & (pl.col("block_key") != "")
    )

    # 3. MEGA-BLOCK PRUNING (Crucial for 1M+ scale)
    # Count occurrences in S23. If a key is too generic, drop it.
    key_counts = s23.group_by(
        ["country", "block_key"]
    ).agg(
        pl.len().alias("n")
    )

    safe_keys = key_counts.filter(
        pl.col("n") <= max_block_size
    )

    s23_safe = s23.join(
        safe_keys,
        on=["country", "block_key"],
        how="inner"
    )

    # 4. Strict country-partitioned Join
    pairs = s1.join(
        s23_safe,
        on=["country", "block_key"],
        how="inner",
        suffix="_s23"
    )

    res = pairs.select([
        pl.col("entity_id").alias("entity_id_s1"),
        pl.col("entity_id_s23"),
        pl.lit(name).alias("source_blocker")
    ])

    if verbose:
        print(
            f"  [{name:^18}] {res.height:>10,} pairs generated "
            f"(pruned blocks > {max_block_size})"
        )

    return res


# ─────────────────────────────────────────────────────────────
# FULL MULTI-PASS RUNNER
# ─────────────────────────────────────────────────────────────

def run_multi_pass_blocking(
    df_s1: pl.DataFrame,
    df_s23: pl.DataFrame,
    max_candidates_per_s1: int = 25,
    batch_size: int = 25_000,
) -> pl.DataFrame:
    """
    Runs all blockers, concatenates, deduplicates, and caps candidates.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    started = time.time()
    batch_results = []
    total_rows = df_s1.height
    total_batches = (total_rows + batch_size - 1) // batch_size

    for batch_number, offset in enumerate(range(0, total_rows, batch_size), 1):
        s1_batch = df_s1.slice(offset, batch_size)
        all_results = [
            generate_candidates(
                s1_batch,
                df_s23,
                blocker["expr"],
                blocker["name"],
                blocker["max_size"],
                verbose=False
            )
            for blocker in BLOCKERS
        ]
        all_pairs = pl.concat(all_results)
        pair_summary = (
            all_pairs
            .group_by(["entity_id_s1", "entity_id_s23"])
            .agg([
                pl.len().alias("num_blockers_agreeing"),
                pl.col("source_blocker").alias("blockers_list")
            ])
            .with_columns(
                pl.col("num_blockers_agreeing")
                .rank("ordinal", descending=True)
                .over("entity_id_s1")
                .alias("rank")
            )
            .filter(pl.col("rank") <= max_candidates_per_s1)
            .drop("rank")
        )
        batch_results.append(pair_summary)
        print(
            f"  [BATCH {batch_number}/{total_batches}] "
            f"Processed {min(offset + batch_size, total_rows):,}/"
            f"{total_rows:,} source-1 rows; "
            f"{pair_summary.height:,} candidates retained"
        )

    if not batch_results:
        return pl.DataFrame(schema={
            "entity_id_s1": pl.String,
            "entity_id_s23": pl.String,
            "num_blockers_agreeing": pl.UInt32,
            "blockers_list": pl.List(pl.String),
        })

    result = pl.concat(batch_results)
    print(
        f"  [CAP]   Top-{max_candidates_per_s1} per S1 → "
        f"{result.height:,} final pairs in {time.time() - started:.1f}s"
    )
    return result


# ─────────────────────────────────────────────────────────────
# RECALL CEILING AUDIT (TRAINING ONLY)
# ─────────────────────────────────────────────────────────────

def audit_recall_ceiling(
    candidates: pl.DataFrame,
    df_gt: pl.DataFrame,
    total_s1: int,
    total_s23: int
):
    """
    Computes the absolute maximum Recall the downstream ML model can achieve.

    If a true match is not in the candidate set,
    the ML model can never find it.
    """

    # 1. Expand comma-separated Ground Truth strings into rows
    gt = (
        df_gt
        .with_columns(
            pl.col("matched_entity_ids").str.split(",")
        )
        .explode("matched_entity_ids")
        .filter(
            pl.col("matched_entity_ids") != ""
        )
        .rename({
            "source1_entity_id": "entity_id_s1",
            "matched_entity_ids": "entity_id_s23"
        })
    )

    total_true = gt.height

    if total_true == 0:
        return

    # 2. Join against candidates
    found = (
        gt.join(
            candidates,
            on=["entity_id_s1", "entity_id_s23"],
            how="inner"
        ).height
    )

    recall = found / total_true

    reduction_ratio = 1.0 - (
        candidates.height / (total_s1 * total_s23)
    )

    print("\n  ── Recall Ceiling Audit (Training Set) ──")
    print(f"  Ground Truth Pairs : {total_true:,}")
    print(f"  Pairs Found        : {found:,}")
    print(
        f"  Recall Ceiling     : {recall:.2%}  "
        "← Maximum possible recall ML can reach"
    )
    print(
        f"  Reduction Ratio    : {reduction_ratio:.6f} "
        f"(Pruned {(total_s1 * total_s23 - candidates.height):,} pairs)"
    )
    print("  ─────────────────────────────────────────")

    if recall < 0.85:
        print(
            "  [WARNING] Recall is below 85%. "
            "You may need to widen max_block_size or add blockers."
        )


# ─────────────────────────────────────────────────────────────
# OUTPUT EXPORT
# ─────────────────────────────────────────────────────────────

def export_candidate_pairs(
    candidates: pl.DataFrame,
    df_s1: pl.DataFrame,
    output_path: str
):
    """
    Format exactly per competition constraints:

    source1_entity_id    candidate_entity_ids
    (Empty list for singletons)
    """

    # 1. Group to comma-separated string
    agg = (
        candidates
        .group_by("entity_id_s1")
        .agg(
            pl.col("entity_id_s23")
            .str.join(",")
            .alias("candidate_entity_ids")
        )
    )

    # 2. Left join against ALL S1 entities to ensure NO row is missing
    # (Singletons get null)
    s1_base = df_s1.select(
        pl.col("entity_id").alias("source1_entity_id")
    )

    final = (
        s1_base
        .join(
            agg,
            left_on="source1_entity_id",
            right_on="entity_id_s1",
            how="left"
        )
        .with_columns(
            pl.col("candidate_entity_ids").fill_null("")
        )
    )

    # 3. Write tab-separated file
    final.write_csv(
        output_path,
        separator="\t"
    )

    print(
        f"  [EXPORT] Wrote {final.height:,} "
        f"candidate pairs to {output_path}"
    )


# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────

def run_blocking(
    data_dict: dict,
    cache_dir: str = "output/cache",
    max_cands: int = 25
) -> dict:
    """
    Execute Candidate Generation on Train and Test sets.
    """

    # NEW:
    # Existing callers pass max_cands=25.
    # Diagnostic analysis showed some true matches losing
    # candidate ranking at the 25-candidate cap.
    # Enforce a minimum of 35 candidates per S1 entity.
    max_cands = max(max_cands, 35)

    print("\n" + "=" * 60)
    print("STEP 3: CANDIDATE GENERATION (BLOCKING)")
    print("=" * 60)

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)

    out_dict = data_dict.copy()

    # ── 1. TRAIN SET ───────────────────────────────────────────

    if "s1_train" in data_dict:
        print("\n[TRAIN SET] Generating Candidates...")

        train_pairs = run_multi_pass_blocking(
            data_dict["s1_train"],
            data_dict["s23_train"],
            max_cands
        )

        # Self-Audit
        if "gt" in data_dict:
            audit_recall_ceiling(
                train_pairs,
                data_dict["gt"],
                data_dict["s1_train"].height,
                data_dict["s23_train"].height
            )

        train_pairs.write_parquet(
            os.path.join(
                cache_dir,
                "candidate_pairs_train.parquet"
            )
        )

        out_dict["train_pairs"] = train_pairs

    # ── 2. TEST SET ────────────────────────────────────────────

    if "s1_test" in data_dict:
        print("\n[TEST SET] Generating Candidates...")

        test_pairs = run_multi_pass_blocking(
            data_dict["s1_test"],
            data_dict["s23_test"],
            max_cands
        )

        # Competition format export — ONLY for Test set
        out_path = "output/candidate_pairs.tsv"

        export_candidate_pairs(
            test_pairs,
            data_dict["s1_test"],
            out_path
        )

        test_pairs.write_parquet(
            os.path.join(
                cache_dir,
                "candidate_pairs_test.parquet"
            )
        )

        out_dict["test_pairs"] = test_pairs

    print("\n" + "=" * 60)
    print("STEP 3 COMPLETE")
    print("=" * 60)

    return out_dict


# ─────────────────────────────────────────────────────────────
# STANDALONE EXECUTION
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import argparse

    from step1_ingestion import run_ingestion
    from step2_normalization import run_normalization

    parser = argparse.ArgumentParser(
        description="Step 3: Blocking"
    )

    parser.add_argument(
        "--train-dir",
        default="dataset/train",
        help="Training data directory"
    )

    parser.add_argument(
        "--test-dir",
        default="dataset/test",
        help="Testing data directory"
    )

    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Ignore caches and re-process"
    )

    args = parser.parse_args()

    # Chain previous steps
    step1 = run_ingestion(
        args.train_dir,
        args.test_dir,
        force_reload=args.force_reload
    )

    step2 = run_normalization(
        step1,
        force_reload=args.force_reload
    )

    # Run Step 3
    step3 = run_blocking(step2)