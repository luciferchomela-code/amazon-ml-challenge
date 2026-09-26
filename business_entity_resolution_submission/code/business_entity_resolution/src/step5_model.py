import os, sys
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

FEATURES = [
    "num_blockers_agreeing", "name_jaro", "name_toksort", "name_tokset",
    "addr_lev", "name_tfidf_cos", "addr_tfidf_cos", "name_len_ratio", "addr_len_ratio",
    "exact_phonetic", "exact_postal", "first_word_name_match", "is_s2"
]


def _load_full_s1_ids(path):
    if not path or not os.path.exists(path):
        return None
    return pl.read_parquet(path)["entity_id"].to_list()


def _load_full_gt_dict(path):
    if not path or not os.path.exists(path):
        return None
    gt_raw = pl.read_parquet(path)
    exploded = (gt_raw.with_columns(pl.col("matched_entity_ids").str.split(","))
                       .explode("matched_entity_ids")
                       .filter(pl.col("matched_entity_ids") != "")
                       .group_by("source1_entity_id")
                       .agg(pl.col("matched_entity_ids").alias("ids")))
    return {s1: set(ids) for s1, ids in zip(exploded["source1_entity_id"].to_list(), exploded["ids"].to_list())}


# FIX #5: include zero-candidate S1 entities in the validation macro-F0.5 pool
def find_optimal_threshold(val_df: pd.DataFrame, extra_singleton_ids: set = None, gt_full_dict: dict = None) -> float:
    if gt_full_dict is not None:
        gt_dict = gt_full_dict
    else:
        gt_dict = val_df[val_df["label"] == 1].groupby("entity_id_s1")["entity_id_s23"].apply(set).to_dict()

    all_val_s1 = set(val_df["entity_id_s1"].unique())
    if extra_singleton_ids:
        all_val_s1 |= set(extra_singleton_ids)
    all_val_s1 = list(all_val_s1)

    best_t, best_score = 0.50, 0.0
    print(f"  [TUNING] Sweeping over {len(all_val_s1):,} S1 entities "
          f"({len(extra_singleton_ids) if extra_singleton_ids else 0:,} zero-candidate included)...")

    for t in np.arange(0.40, 0.98, 0.02):
        preds = val_df[val_df["prob"] >= t].groupby("entity_id_s1")["entity_id_s23"].apply(set).to_dict()
        scores = []
        for s1 in all_val_s1:
            pred_set = preds.get(s1, set())
            true_set = gt_dict.get(s1, set())
            if not pred_set and not true_set:
                scores.append(1.0)
            elif not pred_set or not true_set:
                scores.append(0.0)
            else:
                tp = len(pred_set & true_set)
                prec, rec = tp / len(pred_set), tp / len(true_set)
                scores.append((1.25 * prec * rec) / (0.25 * prec + rec) if prec + rec > 0 else 0.0)
        macro_f05 = sum(scores) / len(scores)
        if macro_f05 > best_score:
            best_score, best_t = macro_f05, t

    print(f"  [TUNING] Best Threshold: {best_t:.2f} (Val F_0.5: {best_score:.4f})")
    return best_t


def train_model(train_features_dir: str, s1_train_path: str = None, gt_path: str = None):
    print("STEP 5A: TRAINING")
    if not os.path.exists(train_features_dir):
        sys.exit(f"[FATAL] Features not found: {train_features_dir}")

    import glob
    import hashlib
    chunk_files = sorted(glob.glob(os.path.join(train_features_dir, "*.parquet")))
    if not chunk_files:
        if os.path.isfile(train_features_dir):
            chunk_files = [train_features_dir]
        else:
            sys.exit(f"[FATAL] No chunks found in: {train_features_dir}")

    val_sets = []
    booster = None
    candidate_ids = set()

    for i, chunk_file in enumerate(chunk_files):
        df = pl.read_parquet(chunk_file).to_pandas()
        df[FEATURES] = df[FEATURES].fillna(0)
        candidate_ids.update(df["entity_id_s1"].unique())

        def is_val(eid):
            return int(hashlib.md5(eid.encode('utf-8')).hexdigest(), 16) % 100 < 15

        mask = df["entity_id_s1"].apply(is_val)
        val_chunk = df[mask]
        train_chunk = df[~mask]

        if len(val_chunk) > 0:
            val_sets.append(val_chunk)

        if len(train_chunk) == 0:
            continue

        n_pos = train_chunk["label"].sum()
        spw = (len(train_chunk) - n_pos) / max(1, n_pos)

        train_data = lgb.Dataset(train_chunk[FEATURES], label=train_chunk["label"])

        params = {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "scale_pos_weight": spw,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }

        booster = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            init_model=booster,
            keep_training_booster=True
        )
        print(f"  [TRAIN] Chunk {i+1}/{len(chunk_files)} trained with spw={spw:.2f}")

    if not val_sets:
        sys.exit("[FATAL] Validation set is empty!")
        
    val_set = pd.concat(val_sets, ignore_index=True)
    print(f"  [TRAIN] Full Val set size: {len(val_set):,}")
    
    fi = pd.DataFrame({"Feature": FEATURES, "Importance": booster.feature_importance()}).sort_values("Importance", ascending=False)
    print(fi.to_string(index=False))

    val_set = val_set.copy()
    val_set["prob"] = booster.predict(val_set[FEATURES])

    extra_singleton_ids = set()
    full_s1_ids = _load_full_s1_ids(s1_train_path)
    gt_full_dict = _load_full_gt_dict(gt_path)

    if full_s1_ids is not None:
        zero_cand_ids = [e for e in full_s1_ids if e not in candidate_ids]
        if zero_cand_ids:
            rng = np.random.RandomState(42)
            n_val_zero = min(len(zero_cand_ids), max(1, int(len(zero_cand_ids) * 0.15)))
            chosen = rng.choice(len(zero_cand_ids), size=n_val_zero, replace=False)
            extra_singleton_ids = {zero_cand_ids[i] for i in chosen}
            print(f"  [DATA] {len(zero_cand_ids):,} zero-candidate S1 entities; "
                  f"{len(extra_singleton_ids):,} sampled into validation pool")
    else:
        print("  [WARN] s1_train_path not provided — val F_0.5 covers candidate-bearing entities only")

    best_thresh = find_optimal_threshold(val_set, extra_singleton_ids, gt_full_dict)
    return booster, best_thresh


def predict_and_export(clf, threshold, test_features_dir, df_s1_test, output_path):
    print("STEP 5B: EXPORT")
    import glob
    chunk_files = sorted(glob.glob(os.path.join(test_features_dir, "*.parquet")))
    if not chunk_files:
        if os.path.isfile(test_features_dir):
            chunk_files = [test_features_dir]
        else:
            sys.exit(f"[FATAL] No chunks found in: {test_features_dir}")

    matches_list = []
    
    for i, chunk_file in enumerate(chunk_files):
        df = pl.read_parquet(chunk_file).to_pandas()
        df[FEATURES] = df[FEATURES].fillna(0)
        df["prob"] = clf.predict(df[FEATURES])
        matches = df[df["prob"] >= threshold]
        matches_list.append(pl.DataFrame(matches[["entity_id_s1", "entity_id_s23"]]))
        print(f"  [EXPORT] Chunk {i+1}/{len(chunk_files)} processed.")

    matches_pl = pl.concat(matches_list) if matches_list else pl.DataFrame({"entity_id_s1": [], "entity_id_s23": []}, schema={"entity_id_s1": pl.Utf8, "entity_id_s23": pl.Utf8})
    agg = matches_pl.group_by("entity_id_s1").agg(pl.col("entity_id_s23").str.join(",").alias("matched_entity_ids"))
    s1_base = df_s1_test.select(pl.col("entity_id").alias("source1_entity_id"))
    final = s1_base.join(agg, left_on="source1_entity_id", right_on="entity_id_s1", how="left") \
                   .with_columns(pl.col("matched_entity_ids").fill_null(""))
    final.write_csv(output_path, separator="\t")
    n_single = (final["matched_entity_ids"] == "").sum()
    print(f"  [EXPORT] {final.height:,} rows, {n_single:,} singletons -> {output_path}")


if __name__ == "__main__":
    import argparse
    from step1_ingestion import run_ingestion

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--cache-dir", default="output/cache")
    args = parser.parse_args()

    train_feat_path = os.path.join(args.cache_dir, "features_train")
    test_feat_path = os.path.join(args.cache_dir, "features_test")
    s1_train_cache = os.path.join(args.cache_dir, "train_source1.parquet")
    gt_cache = os.path.join(args.cache_dir, "train_ground_truth.parquet")

    model, best_thresh = train_model(train_feat_path, s1_train_path=s1_train_cache, gt_path=gt_cache)
    step1_data = run_ingestion(args.train_dir, args.test_dir, force_reload=False, verbose=False)
    predict_and_export(model, best_thresh, test_feat_path, step1_data["s1_test"], "output/matching_results.tsv")
