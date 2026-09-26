"""
diagnose_blocking_misses.py
Finds every GT pair NOT in candidate_pairs_train.parquet and shows, per pair,
which of your blocking keys matched and which didn't — the actual reason
each miss happened, using your real data.

Run: python diagnose_blocking_misses.py --cache-dir output/cache
"""
import argparse
import sys
from pathlib import Path

import polars as pl

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from step3_blocking import BLOCKERS


def explode_gt(gt):
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
              .explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids") != "")
              .rename({"source1_entity_id": "entity_id_s1", "matched_entity_ids": "entity_id_s23"}))


def main(cache_dir):
    s1 = pl.read_parquet(f"{cache_dir}/s1_train_normalized.parquet")
    s23 = pl.read_parquet(f"{cache_dir}/s23_train_normalized.parquet")
    cands = pl.read_parquet(f"{cache_dir}/candidate_pairs_train.parquet")
    gt = pl.read_parquet(f"{cache_dir}/train_ground_truth.parquet")

    gt_pairs = explode_gt(gt)
    missed = gt_pairs.join(cands.select(["entity_id_s1", "entity_id_s23"]),
                            on=["entity_id_s1", "entity_id_s23"], how="anti")
    print(f"Total GT pairs: {gt_pairs.height:,} | Missed by blocking: {missed.height:,}")

    s1_lookup = {r["entity_id"]: r for r in s1.to_dicts()}
    s23_lookup = {r["entity_id"]: r for r in s23.to_dicts()}

    rows = []
    for m in missed.to_dicts():
        a, b = s1_lookup.get(m["entity_id_s1"]), s23_lookup.get(m["entity_id_s23"])
        if a is None or b is None:
            continue
        passed, failed = [], []
        for blk in BLOCKERS:
            ka = pl.DataFrame([a]).with_columns(blk["expr"].alias("k"))["k"][0]
            kb = pl.DataFrame([b]).with_columns(blk["expr"].alias("k"))["k"][0]
            (passed if (ka and kb and ka == kb) else failed).append(blk["name"])
        rows.append({
            "s1_id": m["entity_id_s1"], "s1_name": a["business_name"], "s1_addr": a["business_address"],
            "s23_id": m["entity_id_s23"], "s23_name": b["business_name"], "s23_addr": b["business_address"],
            "country_match": a["country"] == b["country"],
            "blockers_passed": ";".join(passed) or "NONE",
            "blockers_failed": ";".join(failed),
        })

    out = pl.DataFrame(rows)
    out.write_csv("output/missed_matches_diagnostic.tsv", separator="\t")
    with pl.Config(fmt_str_lengths=200, tbl_width_chars=250):
        print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="output/cache")
    main(p.parse_args().cache_dir)