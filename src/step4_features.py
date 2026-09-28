import os, time, pickle, sys
import multiprocessing as mp
from typing import Tuple, List

import polars as pl
import numpy as np
from rapidfuzz import fuzz, distance as rfd
from sklearn.feature_extraction.text import TfidfVectorizer


def fuzzy_batch(args):
    names1, names2, addrs1, addrs2 = args
    res_name_jaro, res_name_toksort, res_name_tokset, res_addr_lev = [], [], [], []
    res_name_partial, res_addr_partial = [], []
    jw_sim = rfd.JaroWinkler.normalized_similarity
    ts_ratio = fuzz.token_sort_ratio
    tset_ratio = fuzz.token_set_ratio
    partial = fuzz.partial_ratio
    lev_sim = rfd.Levenshtein.normalized_similarity
    for n1, n2, a1, a2 in zip(names1, names2, addrs1, addrs2):
        n1, n2, a1, a2 = n1 or "", n2 or "", a1 or "", a2 or ""
        res_name_jaro.append(jw_sim(n1, n2))
        res_name_toksort.append(ts_ratio(n1, n2) / 100.0)
        res_name_tokset.append(tset_ratio(n1, n2) / 100.0)
        res_name_partial.append(partial(n1, n2) / 100.0)
        res_addr_lev.append(lev_sim(a1, a2))
        res_addr_partial.append(partial(a1, a2) / 100.0)
    return res_name_jaro, res_name_toksort, res_name_tokset, res_addr_lev, res_name_partial, res_addr_partial


def add_rapidfuzz_features(df: pl.DataFrame, n_workers: int = None) -> pl.DataFrame:
    n_workers = n_workers or max(1, min(8, mp.cpu_count() - 1))
    print(f"  [FUZZY] {df.height:,} pairs ({n_workers} cores)...")
    start = time.time()
    names1 = df["clean_name_s1"].to_list()
    names2 = df["clean_name_s23"].to_list()
    addrs1 = df["clean_address_s1"].to_list()
    addrs2 = df["clean_address_s23"].to_list()
    if n_workers == 1:
        results = [fuzzy_batch((names1, names2, addrs1, addrs2))]
    else:
        bs = max(1, len(names1) // n_workers)
        batches = [(names1[i:i+bs], names2[i:i+bs], addrs1[i:i+bs], addrs2[i:i+bs])
                   for i in range(0, len(names1), bs)]
        with mp.Pool(n_workers) as pool:
            results = pool.map(fuzzy_batch, batches)
    jaro, toksort, tokset, lev, name_partial, addr_partial = [], [], [], [], [], []
    for r in results:
        jaro.extend(r[0]); toksort.extend(r[1]); tokset.extend(r[2]); lev.extend(r[3]); name_partial.extend(r[4]); addr_partial.extend(r[5])
    df = df.with_columns([
        pl.Series("name_jaro", jaro, dtype=pl.Float32),
        pl.Series("name_toksort", toksort, dtype=pl.Float32),
        pl.Series("name_tokset", tokset, dtype=pl.Float32),
        pl.Series("name_partial", name_partial, dtype=pl.Float32),
        pl.Series("addr_lev", lev, dtype=pl.Float32),
        pl.Series("addr_partial", addr_partial, dtype=pl.Float32),
    ])
    print(f"  [FUZZY] {time.time()-start:.1f}s")
    return df


# FIX #4: fit TF-IDF once on pooled train+test text, cache, and reuse (transform-only)
def fit_tfidf_vectorizers(data_dict: dict, cache_dir: str = "output/cache") -> dict:
    name_cache = os.path.join(cache_dir, "tfidf_name_vectorizer.pkl")
    addr_cache = os.path.join(cache_dir, "tfidf_addr_vectorizer.pkl")
    if os.path.exists(name_cache) and os.path.exists(addr_cache):
        print("  [TF-IDF] Loading cached vectorizers...")
        with open(name_cache, "rb") as f: name_vec = pickle.load(f)
        with open(addr_cache, "rb") as f: addr_vec = pickle.load(f)
        return {"name": name_vec, "addr": addr_vec}

    print("  [TF-IDF] Fitting global vectorizers once...")
    frames = [data_dict[k] for k in ["s1_train", "s23_train", "s1_test", "s23_test"] if k in data_dict]
    all_names = pl.concat([f.select("clean_name") for f in frames])["clean_name"].to_list()
    all_addrs = pl.concat([f.select("clean_address") for f in frames])["clean_address"].to_list()
    name_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=150_000, dtype=np.float32).fit(all_names)
    addr_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=150_000, dtype=np.float32).fit(all_addrs)
    os.makedirs(cache_dir, exist_ok=True)
    with open(name_cache, "wb") as f: pickle.dump(name_vec, f)
    with open(addr_cache, "wb") as f: pickle.dump(addr_vec, f)
    return {"name": name_vec, "addr": addr_vec}


def compute_tfidf_cosine_aligned(v1, v2) -> np.ndarray:
    # v1 and v2 are aligned sparse matrices (same number of rows)
    # Row-wise dot product: sum(A * B, axis=1)
    return np.asarray(v1.multiply(v2).sum(axis=1)).flatten()


def add_polars_features(df: pl.DataFrame) -> pl.DataFrame:
    max_name_len = pl.max_horizontal(pl.col("clean_name_s1").str.len_chars(), pl.col("clean_name_s23").str.len_chars())
    safe_max_name = pl.when(max_name_len == 0).then(1).otherwise(max_name_len)
    max_addr_len = pl.max_horizontal(pl.col("clean_address_s1").str.len_chars(), pl.col("clean_address_s23").str.len_chars())
    safe_max_addr = pl.when(max_addr_len == 0).then(1).otherwise(max_addr_len)
    return df.with_columns([
        (pl.min_horizontal(pl.col("clean_name_s1").str.len_chars(), pl.col("clean_name_s23").str.len_chars()) / safe_max_name)
            .cast(pl.Float32).alias("name_len_ratio"),
        (pl.col("clean_name_s1").str.len_chars() - pl.col("clean_name_s23").str.len_chars()).abs().cast(pl.Float32).alias("name_len_diff"),
        (pl.min_horizontal(pl.col("clean_address_s1").str.len_chars(), pl.col("clean_address_s23").str.len_chars()) / safe_max_addr)
            .cast(pl.Float32).alias("addr_len_ratio"),
        (pl.col("clean_address_s1").str.len_chars() - pl.col("clean_address_s23").str.len_chars()).abs().cast(pl.Float32).alias("addr_len_diff"),
        (pl.col("clean_address_s1").str.replace_all(r"[^\d]", "") == pl.col("clean_address_s23").str.replace_all(r"[^\d]", "")).cast(pl.Int8).alias("exact_addr_numbers"),
        # FIX #3: guard "" == "" false positives
        ((pl.col("postal_code_s1") != "") & (pl.col("postal_code_s23") != "") &
         (pl.col("postal_code_s1") == pl.col("postal_code_s23"))).cast(pl.Int8).alias("exact_postal"),
        ((pl.col("phonetic_name_s1") != "") & (pl.col("phonetic_name_s23") != "") &
         (pl.col("phonetic_name_s1") == pl.col("phonetic_name_s23"))).cast(pl.Int8).alias("exact_phonetic"),
        (pl.col("clean_name_s1").str.split(" ").list.first() == pl.col("clean_name_s23").str.split(" ").list.first())
            .cast(pl.Int8).alias("first_word_name_match"),
        pl.col("entity_id_s23").str.starts_with("S2-").cast(pl.Int8).alias("is_s2")
    ])


def process_features_to_disk(pairs, df_s1, df_s23, label, name_vec, addr_vec, out_dir, df_gt=None):
    print(f"\n[{label}] {pairs.height:,} pairs...")
    os.makedirs(out_dir, exist_ok=True)

    cols = ["entity_id", "clean_name", "clean_address", "phonetic_name", "postal_code"]
    chunk_size = 2_000_000
    num_chunks = (pairs.height + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        print(f"[{label}] Processing chunk {i+1}/{num_chunks}...")
        import time
        start_chunk = time.time()
        chunk = pairs.slice(i * chunk_size, chunk_size)
        
        chunk = chunk.join(df_s1.select(cols), left_on="entity_id_s1", right_on="entity_id", how="left") \
                     .rename({c: f"{c}_s1" for c in cols if c != "entity_id"})
        chunk = chunk.join(df_s23.select(cols), left_on="entity_id_s23", right_on="entity_id", how="left") \
                     .rename({c: f"{c}_s23" for c in cols if c != "entity_id"})
        
        chunk = add_polars_features(chunk)
        
        print(f"  [{label}] Computing TF-IDF for chunk...")
        t0 = time.time()
        v_s1_name = name_vec.transform(chunk["clean_name_s1"].to_list())
        v_s23_name = name_vec.transform(chunk["clean_name_s23"].to_list())
        name_cos = compute_tfidf_cosine_aligned(v_s1_name, v_s23_name)
        
        v_s1_addr = addr_vec.transform(chunk["clean_address_s1"].to_list())
        v_s23_addr = addr_vec.transform(chunk["clean_address_s23"].to_list())
        addr_cos = compute_tfidf_cosine_aligned(v_s1_addr, v_s23_addr)
        print(f"  [{label}] TF-IDF took {time.time()-t0:.1f}s")
        
        chunk = chunk.with_columns([
            pl.Series("name_tfidf_cos", name_cos, dtype=pl.Float32),
            pl.Series("addr_tfidf_cos", addr_cos, dtype=pl.Float32),
        ])
        
        chunk = add_rapidfuzz_features(chunk)
        
        drop_cols = [c for c in chunk.columns if c.endswith("_s1") or c.endswith("_s23")]
        drop_cols.remove("entity_id_s1"); drop_cols.remove("entity_id_s23")
        chunk = chunk.drop(drop_cols)
        
        if df_gt is not None:
            chunk = add_ground_truth_labels(chunk, df_gt)
            
        chunk_path = os.path.join(out_dir, f"chunk_{i:04d}.parquet")
        chunk.write_parquet(chunk_path)
        print(f"[{label}] Wrote chunk {i+1} to {chunk_path}")


def add_ground_truth_labels(pairs: pl.DataFrame, df_gt: pl.DataFrame) -> pl.DataFrame:
    gt = (df_gt.with_columns(pl.col("matched_entity_ids").str.split(","))
               .explode("matched_entity_ids")
               .filter(pl.col("matched_entity_ids") != "")
               .rename({"source1_entity_id": "entity_id_s1", "matched_entity_ids": "entity_id_s23"})
               .with_columns(pl.lit(1).cast(pl.Int8).alias("label")))
    pairs = pairs.join(gt, on=["entity_id_s1", "entity_id_s23"], how="left")
    pairs = pairs.with_columns(pl.col("label").fill_null(0))
    pos = pairs["label"].sum(); neg = pairs.height - pos
    print(f"  [LABELS] {pos:,} pos | {neg:,} neg (1:{neg/max(1,pos):.1f})")
    return pairs


def run_feature_engineering(data_dict: dict, cache_dir: str = "output/cache", force_reload: bool = False) -> dict:
    print("STEP 4: FEATURES")
    out_dict = data_dict.copy()
    train_dir = os.path.join(cache_dir, "features_train")
    test_dir = os.path.join(cache_dir, "features_test")
    need_train = "train_pairs" in data_dict and (force_reload or not os.path.exists(train_dir))
    need_test = "test_pairs" in data_dict and (force_reload or not os.path.exists(test_dir))

    name_vec = addr_vec = None
    if need_train or need_test:
        vecs = fit_tfidf_vectorizers(data_dict, cache_dir=cache_dir)
        name_vec, addr_vec = vecs["name"], vecs["addr"]

    if "train_pairs" in data_dict:
        if os.path.exists(train_dir) and not force_reload:
            print("  [FEATURES] Using cached features_train directory...")
        else:
            process_features_to_disk(data_dict["train_pairs"], data_dict["s1_train"], data_dict["s23_train"],
                                     "TRAIN", name_vec, addr_vec, train_dir, df_gt=data_dict["gt"])
        out_dict["train_features_dir"] = train_dir

    if "test_pairs" in data_dict:
        if os.path.exists(test_dir) and not force_reload:
            print("  [FEATURES] Using cached features_test directory...")
        else:
            process_features_to_disk(data_dict["test_pairs"], data_dict["s1_test"], data_dict["s23_test"],
                                     "TEST", name_vec, addr_vec, test_dir)
        out_dict["test_features_dir"] = test_dir

    print("STEP 4 COMPLETE")
    return out_dict


if __name__ == "__main__":
    import argparse
    from step1_ingestion import run_ingestion
    from step2_normalization import run_normalization
    from step3_blocking import run_blocking

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--force-reload", action="store_true")
    args = parser.parse_args()

    d = run_ingestion(args.train_dir, args.test_dir, force_reload=args.force_reload)
    d = run_normalization(d, force_reload=args.force_reload)
    d = run_blocking(d, max_cands=25)
    d = run_feature_engineering(d, force_reload=args.force_reload)
