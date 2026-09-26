# Business Entity Resolution — Complete In-Depth Pipeline Plan
## Scale: 1M+ Records Per Sour. ce | Metric: F_0.5 | No External Data

---

## Hard Constraints (Never Violate These)

| Constraint | Impact |
|---|---|
| > 1M records per source | No O(N²). No Python loops over full data. Polars + parallel everywhere |
| F_0.5 metric | Precision is 2× more valuable than recall. Every design decision biases toward precision |
| Singletons scored | Predicting empty list correctly = 1.0. Predicting any wrong match = 0.0. Never force a match |
| Country is open-set | France appears only in test. Never one-hot encode. Always string equality comparison |
| Model ≤ 8B params, MIT/Apache 2.0 | Use `all-MiniLM-L6-v2` (22M, Apache 2.0) + LightGBM (MIT) |
| No external data | All signals derived only from the provided TSV files |
| Both TSV outputs required | `matching_results.tsv` (scored) + `candidate_pairs.tsv` (audited) |
| Every S1 entity must appear | Left join against full S1 list at export — singletons get empty string |

---

## Step 1: Data Ingestion & Schema Alignment

### Objective
Load all 6 TSV files (3 train + 3 test) into memory-efficient, correctly-typed structures without silent data corruption.

### Why This Step Can Go Wrong
Reading `.tsv` without `sep="\t"` silently produces a single column containing the entire row. Business addresses and ID lists both contain commas — this will destroy your data silently.

### Implementation

```python
import polars as pl

# Use Polars — 1M rows read in ~2 seconds vs ~15 seconds with Pandas
# infer_schema_length=0 forces all columns to string — prevents NaN float type
# corrupting downstream string operations

def load_source(path):
    return pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=0,    # All columns as string — critical
        null_values=["", "NA", "N/A", "null", "NULL"],
        truncate_ragged_lines=True
    )

df_s1_train = load_source("dataset/train/train_source1.tsv")
df_s2_train = load_source("dataset/train/train_source2.tsv")
df_s3_train = load_source("dataset/train/train_source3.tsv")
df_gt       = load_source("dataset/train/train_ground_truth.tsv")

df_s1_test  = load_source("dataset/test/test_source1.tsv")
df_s2_test  = load_source("dataset/test/test_source2.tsv")
df_s3_test  = load_source("dataset/test/test_source3.tsv")

# Combine S2 + S3 into one pool — all blockers query a single unified 2M-row pool
df_s23_train = pl.concat([df_s2_train, df_s3_train], how="vertical")
df_s23_test  = pl.concat([df_s2_test,  df_s3_test],  how="vertical")
```

### Source Tagging (Vectorized — No Python Loop)
```python
# Vectorized string slice — extracts "S1", "S2", "S3" from entity_id in one pass
# Never use .apply(lambda x: x.split("-")[0]) — that loops 3M times in Python
df_s1_train  = df_s1_train.with_columns( pl.col("entity_id").str.slice(0, 2).alias("source_tag"))
df_s23_train = df_s23_train.with_columns(pl.col("entity_id").str.slice(0, 2).alias("source_tag"))
```

### Missing Value Safety
```python
# Fill NaN in all text columns immediately — downstream string ops crash on None
text_cols = ["business_name", "business_address", "country"]
df_s1_train  = df_s1_train.with_columns( [pl.col(c).fill_null("") for c in text_cols])
df_s23_train = df_s23_train.with_columns([pl.col(c).fill_null("") for c in text_cols])
```

### Early Validation
```python
# Sanity check: all S1 IDs start with "S1-", all S23 start with "S2-" or "S3-"
assert df_s1_train.filter(~pl.col("entity_id").str.starts_with("S1-")).is_empty()
assert df_s23_train.filter(
    ~pl.col("entity_id").str.starts_with("S2-") &
    ~pl.col("entity_id").str.starts_with("S3-")
).is_empty()
print(f"S1 train: {len(df_s1_train):,} | S23 train: {len(df_s23_train):,}")
```

**Outputs:** `df_s1_train`, `df_s23_train`, `df_s1_test`, `df_s23_test` — all Polars DataFrames with uniform schema and no nulls in text columns.

---

## Step 2: Text Normalization & Standardization

### Objective
Reduce surface-level noise so that two records describing the same business look as similar as possible before any comparison begins. Good normalization eliminates noise cheaply — ML doesn't have to learn to ignore "Corp" vs "Corporation".

### 2a. Global Cleansing (One Compiled Regex Pass)

```python
import re

# Compile once — used for all 3M rows
# Matches: any non-alphanumeric char (except space) OR multiple consecutive spaces
CLEAN_PATTERN = re.compile(r"[^a-z0-9\s]|\s+")

def global_clean_polars(df, col):
    """Vectorized Polars string cleaning — pure C SIMD, no Python loop."""
    return (df
        .with_columns(pl.col(col).str.to_lowercase().alias(col))
        .with_columns(pl.col(col).str.replace_all(r"[^a-z0-9\s]", " ", literal=False).alias(col))
        .with_columns(pl.col(col).str.replace_all(r"\s+", " ", literal=False).alias(col))
        .with_columns(pl.col(col).str.strip_chars().alias(col))
    )

for df_ref in [df_s1_train, df_s23_train]:
    df_ref = global_clean_polars(df_ref, "business_name")
    df_ref = global_clean_polars(df_ref, "business_address")
```

### 2b. Abbreviation Expansion (FlashText — Aho-Corasick O(N) Regardless of Dictionary Size)

```python
# pip install flashtext
from flashtext import KeywordProcessor

# Business name suffixes
name_kp = KeywordProcessor()
name_kp.add_keywords_from_dict({
    "corporation": ["corp", "corpn", "crp"],
    "incorporated": ["inc", "incorp"],
    "private": ["pvt", "prvt"],
    "limited": ["ltd", "lmtd"],
    "company": ["co", "coy"],
    "and": ["&"],
    "industries": ["inds", "ind"],
    "enterprises": ["ent", "entp"],
    "services": ["svc", "svcs"],
    "international": ["intl", "int"],
})

# Address abbreviations
addr_kp = KeywordProcessor()
addr_kp.add_keywords_from_dict({
    "street": ["st", "str"],
    "road": ["rd"],
    "avenue": ["ave"],
    "boulevard": ["blvd"],
    "north": ["n"],
    "south": ["s"],
    "east": ["e"],
    "west": ["w"],
    "apartment": ["apt"],
    "building": ["bldg"],
    "floor": ["fl"],
})

# Apply — list comprehension on pre-extracted list is faster than Polars apply
# for FlashText since it is a Python-level operation
name_list = df_s23_train["business_name"].to_list()
df_s23_train = df_s23_train.with_columns(
    pl.Series("clean_name", [name_kp.replace_keywords(t) for t in name_list])
)
```

### 2c. Acronym Squeezing (Single Letters Separated by Spaces)

```python
ACRONYM_PATTERN = re.compile(r"(?<=\b[a-z])\s(?=[a-z]\b)")

def squeeze_acronyms(text):
    # "i b m" → "ibm", "a t m" → "atm"
    return ACRONYM_PATTERN.sub("", text)

# Apply with Polars regex — vectorized
df_s23_train = df_s23_train.with_columns(
    pl.col("clean_name").str.replace_all(r"(?<=\b[a-z])\s(?=[a-z]\b)", "", literal=False).alias("clean_name")
)
```

### 2d. Token Sorting (Word-Order Invariance)

```python
# Polars native list ops — no Python apply()
# "sbi atm near main" → "atm main near sbi"
df_s23_train = df_s23_train.with_columns(
    pl.col("clean_name")
      .str.split(" ")
      .list.sort()
      .list.join(" ")
      .alias("sorted_name")
)
df_s1_train = df_s1_train.with_columns(
    pl.col("clean_name")
      .str.split(" ")
      .list.sort()
      .list.join(" ")
      .alias("sorted_name")
)
```

### 2e. Phonetic Hashing (Parallel — NOT a Python For-Loop)

```python
import jellyfish
import multiprocessing as mp

def phonetic_batch(texts):
    """Runs in a separate worker process."""
    return [jellyfish.metaphone(t.split()[0]) if t.strip() else "" for t in texts]

def parallel_phonetic(series, n_workers=None):
    n_workers = n_workers or (mp.cpu_count() - 1)
    texts     = series.to_list()
    batch_sz  = max(1, len(texts) // n_workers)
    batches   = [texts[i:i+batch_sz] for i in range(0, len(texts), batch_sz)]
    with mp.Pool(n_workers) as pool:
        results = pool.map(phonetic_batch, batches)
    return [code for batch in results for code in batch]

df_s1_train  = df_s1_train.with_columns( pl.Series("phonetic_name", parallel_phonetic(df_s1_train["clean_name"])))
df_s23_train = df_s23_train.with_columns(pl.Series("phonetic_name", parallel_phonetic(df_s23_train["clean_name"])))
```

### 2f. Postal Code Extraction (Before Global Clean Strips Digits)

```python
# Extract BEFORE stripping — regex on raw business_address
# US ZIP: 5 digits. India PIN: 6 digits. France: 5 digits (same as US pattern — handled)
POSTAL_RE = r"\b(\d{5}(?:-\d{4})?|\d{6})\b"

df_s1_train  = df_s1_train.with_columns( pl.col("business_address").str.extract(POSTAL_RE, 1).alias("postal_code"))
df_s23_train = df_s23_train.with_columns(pl.col("business_address").str.extract(POSTAL_RE, 1).alias("postal_code"))
```

### 2g. Over-Normalization Guard (S1 Collision Test)

```python
def collision_test(df_s1, col):
    """
    S1 is deduplicated — every row is a distinct real-world business.
    If two S1 rows collapse to the same normalized value, normalization is too aggressive.
    Run after every normalization step.
    """
    collisions = (df_s1
                  .group_by(col)
                  .agg(pl.count().alias("n"))
                  .filter(pl.col("n") > 1))
    if len(collisions) > 0:
        print(f"WARNING [{col}]: {len(collisions)} collision groups — normalization too aggressive!")
        print(collisions.head(5))
    else:
        print(f"OK [{col}]: No S1 collisions.")

collision_test(df_s1_train, "clean_name")
collision_test(df_s1_train, "sorted_name")
```

**Final Columns Available After Step 2:**
`entity_id`, `clean_name`, `sorted_name`, `phonetic_name`, `clean_address`, `postal_code`, `country`, `source_tag`

---

## Step 3: Index Construction

### Objective
Transform S2+S3 into 5 query-ready data structures. Build once, save to disk, reload on re-runs.

### 3a. Hash Index (Polars group_by — Parallel C)

```python
import pickle, os
os.makedirs("output/indexes", exist_ok=True)

# Build blocking keys first
df_s23_train = df_s23_train.with_columns([
    (pl.col("country") + "|" +
     pl.col("sorted_name").str.slice(0, 4).fill_null("") + "|" +
     pl.col("postal_code").fill_null("")).alias("block_key_exact"),

    (pl.col("country") + "|" +
     pl.col("phonetic_name").fill_null("")).alias("block_key_phonetic"),
])

df_s1_train = df_s1_train.with_columns([
    (pl.col("country") + "|" +
     pl.col("sorted_name").str.slice(0, 4).fill_null("") + "|" +
     pl.col("postal_code").fill_null("")).alias("block_key_exact"),

    (pl.col("country") + "|" +
     pl.col("phonetic_name").fill_null("")).alias("block_key_phonetic"),
])

def build_hash_index_polars(df, key_col, id_col="entity_id"):
    grouped = df.group_by(key_col).agg(pl.col(id_col).alias("ids"))
    return dict(zip(grouped[key_col].to_list(), grouped["ids"].to_list()))

idx_exact    = build_hash_index_polars(df_s23_train, "block_key_exact")
idx_phonetic = build_hash_index_polars(df_s23_train, "block_key_phonetic")
idx_postal   = build_hash_index_polars(df_s23_train.filter(pl.col("postal_code").is_not_null()), "postal_code")

with open("output/indexes/hash_exact.pkl",    "wb") as f: pickle.dump(idx_exact, f)
with open("output/indexes/hash_phonetic.pkl", "wb") as f: pickle.dump(idx_phonetic, f)
with open("output/indexes/hash_postal.pkl",   "wb") as f: pickle.dump(idx_postal, f)
print("Hash indexes saved.")
```

### 3b. FAISS IVF-PQ Index (Dense ANN — 1M Scale)

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL    = SentenceTransformer("all-MiniLM-L6-v2")  # 22M params, Apache 2.0
DIM      = 384
NLIST    = 4096    # Voronoi cells
M_PQ     = 48      # Subquantizers (384/48=8 — must divide evenly)
BITS     = 8       # Bits per subquantizer → 16x compression vs FlatIP
NPROBE   = 128     # Cells searched per query — ~3% of index

def make_text(df):
    return (df["clean_name"] + " " +
            df["clean_address"].fill_null("") + " " +
            df["country"].fill_null("")).to_list()

# Encode S2+S3 → memmap (never loads 3 GB into RAM)
s23_texts = make_text(df_s23_train)
N_S23     = len(s23_texts)
emb_s23   = np.memmap("output/indexes/emb_s23.npy", dtype="float32", mode="w+", shape=(N_S23, DIM))

BATCH = 1024
for i in range(0, N_S23, BATCH):
    vecs = MODEL.encode(s23_texts[i:i+BATCH], normalize_embeddings=True, show_progress_bar=False)
    emb_s23[i:i+BATCH] = vecs
    if i % 100_000 == 0: print(f"  Encoded {i:,}/{N_S23:,}")
emb_s23.flush()

# Encode S1 → RAM (1M × 384 × 4 = 1.5 GB)
s1_texts = make_text(df_s1_train)
emb_s1   = MODEL.encode(s1_texts, batch_size=BATCH, normalize_embeddings=True,
                         show_progress_bar=True).astype("float32")
np.save("output/indexes/emb_s1.npy", emb_s1)

# Build IVF-PQ
quantizer   = faiss.IndexFlatIP(DIM)
faiss_index = faiss.IndexIVFPQ(quantizer, DIM, NLIST, M_PQ, BITS)
faiss_index.metric_type = faiss.METRIC_INNER_PRODUCT

# Train on 500k representative S23 samples
train_sample = np.array(emb_s23[:500_000])
faiss.normalize_L2(train_sample)
faiss_index.train(train_sample)

# Add all S23 in batches
for i in range(0, N_S23, 100_000):
    batch = np.array(emb_s23[i:i+100_000])
    faiss.normalize_L2(batch)
    faiss_index.add(batch)
    print(f"  Added {min(i+100_000, N_S23):,}/{N_S23:,}")

faiss_index.nprobe = NPROBE
faiss.write_index(faiss_index, "output/indexes/faiss_ivfpq.index")
print(f"FAISS built: {faiss_index.ntotal:,} vectors | RAM: ~600 MB (vs 3 GB for Flat)")
```

### 3c. Char N-gram Inverted Index (Parallel Sharded Build)

```python
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import math

def build_shard(args):
    ids, texts, n = args
    idx = defaultdict(set)
    for eid, t in zip(ids, texts):
        t = str(t)
        for i in range(len(t) - n + 1):
            idx[t[i:i+n]].add(eid)
    return dict(idx)

def merge_shards(shards):
    merged = defaultdict(set)
    for sh in shards:
        for ng, eids in sh.items():
            merged[ng].update(eids)
    return dict(merged)

ids   = df_s23_train["entity_id"].to_list()
texts = df_s23_train["clean_name"].fill_null("").to_list()
N_W   = mp.cpu_count() - 1
sz    = max(1, len(ids) // N_W)
args  = [(ids[i:i+sz], texts[i:i+sz], 3) for i in range(0, len(ids), sz)]

with ProcessPoolExecutor(max_workers=N_W) as ex:
    shards = list(ex.map(build_shard, args))

inv_index = merge_shards(shards)

# Prune high-frequency n-grams (> 5% of docs = generic, not discriminative)
MAX_DF    = int(0.05 * len(ids))
inv_index = {ng: eids for ng, eids in inv_index.items() if len(eids) <= MAX_DF}

N_DOCS = len(ids)
idf    = {ng: math.log((N_DOCS+1)/(len(eids)+1)) for ng, eids in inv_index.items()}

with open("output/indexes/inv_index.pkl", "wb") as f: pickle.dump(inv_index, f)
with open("output/indexes/idf_map.pkl",   "wb") as f: pickle.dump(idf, f)
print(f"Inverted index: {len(inv_index):,} unique 3-grams")
```

### 3d. Sorted Array (Polars sort — < 2 seconds)

```python
import bisect

df_s23_sorted = df_s23_train.sort("clean_name")
snm_keys = df_s23_sorted["clean_name"].to_list()
snm_ids  = df_s23_sorted["entity_id"].to_list()

with open("output/indexes/snm_keys.pkl", "wb") as f: pickle.dump(snm_keys, f)
with open("output/indexes/snm_ids.pkl",  "wb") as f: pickle.dump(snm_ids, f)
```

---

## Step 4: Blocking / Candidate Generation

### Objective
Generate ≤ 25 candidate pairs per S1 entity by querying all indexes in parallel, then union-merge with provenance metadata.

### 4a. Blocker Execution

```python
all_raw = []  # List of (s1_id, s23_id, score, blocker_name)

# B1: Exact Composite Key — Polars inner join (pure C, no Python loop)
cands_exact = (
    df_s1_train.select(["entity_id", "block_key_exact"])
    .join(
        df_s23_train.select(["entity_id", "block_key_exact"])
                    .rename({"entity_id": "entity_id_s23"}),
        on="block_key_exact", how="inner"
    )
    .select([pl.col("entity_id").alias("entity_id_s1"), "entity_id_s23"])
    .with_columns(pl.lit(1.0).alias("score"), pl.lit("exact").alias("blocker"))
)

# B2: Phonetic Key — Polars inner join
cands_phonetic = (
    df_s1_train.select(["entity_id", "block_key_phonetic"])
    .join(
        df_s23_train.select(["entity_id", "block_key_phonetic"])
                    .rename({"entity_id": "entity_id_s23"}),
        on="block_key_phonetic", how="inner"
    )
    .select([pl.col("entity_id").alias("entity_id_s1"), "entity_id_s23"])
    .with_columns(pl.lit(0.85).alias("score"), pl.lit("phonetic").alias("blocker"))
)

# B3: Postal Code — Polars inner join
df_s1_pc  = df_s1_train.filter(pl.col("postal_code").is_not_null())
df_s23_pc = df_s23_train.filter(pl.col("postal_code").is_not_null())
cands_postal = (
    df_s1_pc.select(["entity_id", "postal_code"])
    .join(
        df_s23_pc.select(["entity_id", "postal_code"])
                 .rename({"entity_id": "entity_id_s23"}),
        on="postal_code", how="inner"
    )
    .select([pl.col("entity_id").alias("entity_id_s1"), "entity_id_s23"])
    .with_columns(pl.lit(0.95).alias("score"), pl.lit("postal").alias("blocker"))
)

# B4: FAISS ANN — batched query
faiss_index = faiss.read_index("output/indexes/faiss_ivfpq.index")
faiss_index.nprobe = NPROBE
s1_ids  = df_s1_train["entity_id"].to_list()
s23_ids = df_s23_train["entity_id"].to_list()
emb_s1  = np.load("output/indexes/emb_s1.npy")

TOP_K = 15
FAISS_THRESH = 0.72
faiss_rows = []

for start in range(0, len(s1_ids), 10_000):
    batch = emb_s1[start:start+10_000].copy()
    faiss.normalize_L2(batch)
    dists, idxs = faiss_index.search(batch, TOP_K)
    for li, (d_row, i_row) in enumerate(zip(dists, idxs)):
        gi = start + li
        for dist, s23_idx in zip(d_row, i_row):
            if s23_idx >= 0 and dist >= FAISS_THRESH:
                faiss_rows.append((s1_ids[gi], s23_ids[s23_idx], float(dist), "faiss"))

cands_faiss = pl.DataFrame(faiss_rows, schema=["entity_id_s1","entity_id_s23","score","blocker"])

# B5: Sorted Neighborhood — bisect O(log N) per query
WINDOW = 5
snm_rows = []
for s1_id, name in zip(s1_ids, df_s1_train["clean_name"].to_list()):
    pos = bisect.bisect_left(snm_keys, name)
    for j in range(max(0, pos-WINDOW), min(len(snm_keys), pos+WINDOW+1)):
        if snm_ids[j] != s1_id:
            snm_rows.append((s1_id, snm_ids[j], 0.5, "snm"))

cands_snm = pl.DataFrame(snm_rows, schema=["entity_id_s1","entity_id_s23","score","blocker"])
```

### 4b. Union Merge + Provenance Enrichment

```python
all_cands = pl.concat([cands_exact, cands_phonetic, cands_postal, cands_faiss, cands_snm])

# Aggregate per unique pair — provenance tracking
pair_summary = (all_cands
    .group_by(["entity_id_s1", "entity_id_s23"])
    .agg([
        pl.count().alias("num_blockers_agreeing"),
        pl.col("score").max().alias("max_score"),
        pl.col("blocker").str.concat("|").alias("blocker_list"),
        pl.col("blocker").map_elements(lambda x: int("exact"   in x), return_dtype=pl.Int8).first().alias("found_by_exact"),
        pl.col("blocker").map_elements(lambda x: int("phonetic" in x), return_dtype=pl.Int8).first().alias("found_by_phonetic"),
        pl.col("blocker").map_elements(lambda x: int("postal"   in x), return_dtype=pl.Int8).first().alias("found_by_postal"),
        pl.col("blocker").map_elements(lambda x: int("faiss"    in x), return_dtype=pl.Int8).first().alias("found_by_embedding"),
    ])
)
```

### 4c. Post-Merge Guards

```python
# Country guard — open-set string equality (works for France automatically)
pair_summary = (pair_summary
    .join(df_s1_train.select(["entity_id","country"]).rename({"entity_id":"entity_id_s1","country":"country_s1"}), on="entity_id_s1")
    .join(df_s23_train.select(["entity_id","country"]).rename({"entity_id":"entity_id_s23","country":"country_s23"}), on="entity_id_s23")
    .filter(pl.col("country_s1") == pl.col("country_s23"))
)

# Self-match guard
pair_summary = pair_summary.filter(~pl.col("entity_id_s23").str.starts_with("S1-"))

# Block explosion cap — at most 25 candidates per S1 entity
MAX_K = 25
pair_summary = (pair_summary
    .sort(["entity_id_s1","num_blockers_agreeing","max_score"], descending=[False,True,True])
    .with_columns(pl.col("entity_id_s1").cum_count().over("entity_id_s1").alias("rank"))
    .filter(pl.col("rank") <= MAX_K)
    .drop("rank")
)

print(f"Final candidate pairs: {len(pair_summary):,}")
```

### 4d. Recall Ceiling Audit (Training Only)

```python
def recall_ceiling_audit(candidates, gt_df):
    # Explode ground truth
    gt = (gt_df
          .with_columns(pl.col("matched_entity_ids").str.split(",").alias("ids"))
          .explode("ids")
          .filter(pl.col("ids").str.strip_chars() != "")
          .rename({"source1_entity_id":"entity_id_s1", "ids":"entity_id_s23"})
          .with_columns(pl.col("entity_id_s23").str.strip_chars())
          .select(["entity_id_s1","entity_id_s23"]))

    merged   = gt.join(candidates.select(["entity_id_s1","entity_id_s23"]),
                        on=["entity_id_s1","entity_id_s23"], how="left", suffix="_cand")
    total    = len(gt)
    found    = merged.filter(pl.col("entity_id_s23_right").is_not_null()).height
    recall   = found / total if total > 0 else 0.0
    rr       = 1 - len(candidates) / (len(df_s1_train) * len(df_s23_train))

    print(f"Recall Ceiling : {recall:.2%} ({found:,}/{total:,})")
    print(f"Reduction Ratio: {rr:.6f}")
    print(f"Total Candidates: {len(candidates):,}")
    assert recall >= 0.90, f"Recall too low ({recall:.2%}). Widen blockers before proceeding."
    return recall

recall_ceiling_audit(pair_summary, df_gt)
```

### 4e. Export `candidate_pairs.tsv`

```python
all_s1 = df_s1_test.select(pl.col("entity_id").alias("source1_entity_id"))
cand_agg = (pair_summary
    .group_by("entity_id_s1")
    .agg(pl.col("entity_id_s23").unique().str.join(",").alias("candidate_entity_ids"))
    .rename({"entity_id_s1":"source1_entity_id"})
)
output_cands = (all_s1
    .join(cand_agg, on="source1_entity_id", how="left")
    .with_columns(pl.col("candidate_entity_ids").fill_null(""))
)
output_cands.write_csv("output/candidate_pairs.tsv", separator="\t")
print(f"candidate_pairs.tsv: {len(output_cands):,} rows | Singletons: {(output_cands['candidate_entity_ids']=='').sum():,}")
```

---

## Step 5: Pairwise Feature Engineering

### Objective
Convert every candidate pair into a ~55-dimensional numerical feature vector. At 25M pairs, every computation must run in C, not Python.

### Setup — Pair Expansion

```python
import pandas as pd  # rapidfuzz works with Pandas/lists
pairs = pair_summary.to_pandas()

# Join both sides
pairs = pairs.merge(
    df_s1_train.select(["entity_id","clean_name","sorted_name","phonetic_name","clean_address","postal_code","country"])
               .to_pandas().add_suffix("_s1").rename(columns={"entity_id_s1":"entity_id_s1"}),
    on="entity_id_s1", how="left"
).merge(
    df_s23_train.select(["entity_id","clean_name","sorted_name","phonetic_name","clean_address","postal_code","country"])
                .to_pandas().add_suffix("_s23").rename(columns={"entity_id_s23":"entity_id_s23"}),
    on="entity_id_s23", how="left"
)
```

### Family 1: Lexical Distances (rapidfuzz.cdist — All CPU Cores)

```python
from rapidfuzz.process import cdist
from rapidfuzz import fuzz, distance as rfd
import numpy as np

CHUNK = 500_000  # Process 500k pairs at a time to control RAM

def add_rapidfuzz_features(pairs, col_s1, col_s2, prefix):
    for start in range(0, len(pairs), CHUNK):
        sl = slice(start, start + CHUNK)
        a  = pairs[col_s1].iloc[sl].fillna("").tolist()
        b  = pairs[col_s2].iloc[sl].fillna("").tolist()
        # cdist computes pairwise — we take the diagonal (pair[i,i])
        pairs.loc[sl, f"{prefix}_lev"]        = np.diag(cdist(a, b, scorer=fuzz.ratio,           workers=-1)) / 100
        pairs.loc[sl, f"{prefix}_jaro"]       = np.diag(cdist(a, b, scorer=rfd.JaroWinkler.normalized_similarity, workers=-1))
        pairs.loc[sl, f"{prefix}_partial"]    = np.diag(cdist(a, b, scorer=fuzz.partial_ratio,    workers=-1)) / 100
        pairs.loc[sl, f"{prefix}_tok_sort"]   = np.diag(cdist(a, b, scorer=fuzz.token_sort_ratio, workers=-1)) / 100
        pairs.loc[sl, f"{prefix}_tok_set"]    = np.diag(cdist(a, b, scorer=fuzz.token_set_ratio,  workers=-1)) / 100

add_rapidfuzz_features(pairs, "clean_name_s1",    "clean_name_s23",    "name")
add_rapidfuzz_features(pairs, "clean_address_s1",  "clean_address_s23",  "addr")
add_rapidfuzz_features(pairs, "sorted_name_s1",    "sorted_name_s23",    "sorted")
```

### Family 2: Token Set Similarities + TF-IDF Cosine

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

# Jaccard — vectorized via Polars set ops on list columns
def jaccard_series(a, b):
    s1 = [set(str(x).split()) for x in a]
    s2 = [set(str(x).split()) for x in b]
    return [(len(x&y)/len(x|y) if x|y else 1.0) for x,y in zip(s1,s2)]

pairs["name_jaccard"] = jaccard_series(pairs["clean_name_s1"], pairs["clean_name_s23"])
pairs["addr_jaccard"] = jaccard_series(pairs["clean_address_s1"], pairs["clean_address_s23"])

# TF-IDF Cosine — sparse row-wise dot product (no dense N×M matrix)
all_names = pd.concat([df_s1_train["clean_name"], df_s23_train["clean_name"]]).fillna("").to_pandas().tolist() if hasattr(df_s1_train["clean_name"], "to_pandas") else []
tfidf_v   = TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=100_000).fit(all_names)

def batch_tfidf_cosine(a_series, b_series, vectorizer, chunk=100_000):
    cosines = []
    a_list, b_list = a_series.fillna("").tolist(), b_series.fillna("").tolist()
    for i in range(0, len(a_list), chunk):
        va = vectorizer.transform(a_list[i:i+chunk])
        vb = vectorizer.transform(b_list[i:i+chunk])
        cosines.extend(np.asarray(va.multiply(vb).sum(axis=1)).flatten())
    return cosines

pairs["name_tfidf_cosine"] = batch_tfidf_cosine(pairs["clean_name_s1"], pairs["clean_name_s23"], tfidf_v)
pairs["addr_tfidf_cosine"] = batch_tfidf_cosine(pairs["clean_address_s1"], pairs["clean_address_s23"], tfidf_v)
```

### Family 3: Phonetic + Structural + Address Features

```python
# Phonetic exact match (vectorized)
pairs["name_phonetic_match"] = (pairs["phonetic_name_s1"] == pairs["phonetic_name_s23"]).astype(int)

# Country match — open-set string equality
pairs["country_match"] = (pairs["country_s1"] == pairs["country_s23"]).astype(int)

# Postal exact + prefix
pairs["postal_exact"]  = (pairs["postal_code_s1"].fillna("X") == pairs["postal_code_s23"].fillna("Y")).astype(int)
pairs["postal_prefix"] = pairs.apply(
    lambda r: int(str(r["postal_code_s1"])[:3] == str(r["postal_code_s23"])[:3])
              if r["postal_code_s1"] and r["postal_code_s23"] else 0, axis=1)

# First token match
pairs["first_tok_match"] = [
    int(str(a).split()[0] == str(b).split()[0])
    if str(a).split() and str(b).split() else 0
    for a, b in zip(pairs["clean_name_s1"], pairs["clean_name_s23"])
]

# Name length ratio
pairs["name_len_ratio"] = pairs.apply(
    lambda r: min(len(str(r["clean_name_s1"])), len(str(r["clean_name_s23"]))) /
              max(len(str(r["clean_name_s1"])), len(str(r["clean_name_s23"])), 1), axis=1)

# Source pair type
pairs["is_s2_pair"] = pairs["entity_id_s23"].str.startswith("S2-").astype(int)
```

### Family 4: Embedding Cosine (Reuse Pre-computed Vectors)

```python
s1_id_map  = {eid: i for i, eid in enumerate(df_s1_train["entity_id"].to_list())}
s23_id_map = {eid: i for i, eid in enumerate(df_s23_train["entity_id"].to_list())}
emb_s1_arr  = np.load("output/indexes/emb_s1.npy")
emb_s23_arr = np.memmap("output/indexes/emb_s23.npy", dtype="float32", mode="r", shape=(len(df_s23_train), DIM))

def batch_emb_cosine(pair_df, chunk=100_000):
    cosines = []
    for start in range(0, len(pair_df), chunk):
        sl   = slice(start, start+chunk)
        v1   = emb_s1_arr[ [s1_id_map[e]  for e in pair_df["entity_id_s1"].iloc[sl]]]
        v2   = emb_s23_arr[[s23_id_map[e] for e in pair_df["entity_id_s23"].iloc[sl]]]
        cosines.extend((v1 * v2).sum(axis=1).tolist())
    return cosines

pairs["emb_cosine"] = batch_emb_cosine(pairs)
```

**Final Feature Matrix:**
```python
FEATURES = [
    "name_lev","name_jaro","name_partial","name_tok_sort","name_tok_set",
    "addr_lev","addr_jaro","addr_partial","addr_tok_sort","addr_tok_set",
    "sorted_lev","sorted_jaro",
    "name_jaccard","addr_jaccard","name_tfidf_cosine","addr_tfidf_cosine",
    "name_phonetic_match","country_match","postal_exact","postal_prefix",
    "first_tok_match","name_len_ratio","is_s2_pair",
    "emb_cosine",
    "num_blockers_agreeing","max_score",
    "found_by_exact","found_by_phonetic","found_by_postal","found_by_embedding",
]

X = pairs[FEATURES].fillna(0).astype("float32").values
```

---

## Step 6: ML Matching Model

### Objective
Train a binary classifier to distinguish true matches from false candidates. Optimize for F_0.5, not accuracy.

### Label Generation

```python
# Explode ground truth → set of true (s1_id, s23_id) pairs
gt_set = set()
for row in df_gt.iter_rows(named=True):
    s1_id = row["source1_entity_id"]
    for s23_id in str(row["matched_entity_ids"]).split(","):
        s23_id = s23_id.strip()
        if s23_id:
            gt_set.add((s1_id, s23_id))

pairs["label"] = pairs.apply(lambda r: int((r["entity_id_s1"], r["entity_id_s23"]) in gt_set), axis=1)
y = pairs["label"].values
print(f"Positive rate: {y.mean():.4%}")  # Expect << 1% — severe class imbalance
```

### Entity-Level Train/Val Split (No Data Leakage)

```python
import numpy as np

all_s1_ids = pairs["entity_id_s1"].unique()
np.random.seed(42)
val_s1_ids = set(np.random.choice(all_s1_ids, size=int(0.2 * len(all_s1_ids)), replace=False))

train_mask = ~pairs["entity_id_s1"].isin(val_s1_ids)
val_mask   = pairs["entity_id_s1"].isin(val_s1_ids)

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
```

### LightGBM Training

```python
import lightgbm as lgb

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=-1,
    scale_pos_weight=n_neg / n_pos,   # Handle severe class imbalance
    colsample_bytree=0.8,
    subsample=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    verbose=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False),
               lgb.log_evaluation(100)],
)

# Feature importance audit
fi = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
print(fi.head(10))
```

---

## Step 7: Decision Thresholding & Singleton Handling

### Objective
Find the threshold that maximizes macro-averaged F_0.5 on the validation set. Never use 0.5 as default.

```python
val_probs = model.predict_proba(X_val)[:, 1]
val_pairs = pairs[val_mask][["entity_id_s1","entity_id_s23","label"]].copy()
val_pairs["prob"] = val_probs

def compute_macro_f05(df_pairs, threshold, gt_set, all_s1_val):
    preds = df_pairs[df_pairs["prob"] >= threshold].groupby("entity_id_s1")["entity_id_s23"].apply(set)
    scores = []
    for s1_id in all_s1_val:
        pred = preds.get(s1_id, set())
        true = {s23 for s1, s23 in gt_set if s1 == s1_id}
        if not pred and not true:
            scores.append(1.0)
        elif not pred or not true:
            scores.append(0.0)
        else:
            tp   = len(pred & true)
            prec = tp / len(pred)
            rec  = tp / len(true)
            f05  = (1.25 * prec * rec) / (0.25 * prec + rec) if prec + rec > 0 else 0.0
            scores.append(f05)
    return sum(scores) / len(scores)

best_thresh, best_f05 = 0.5, 0.0
all_s1_val = list(val_s1_ids)

for thresh in np.arange(0.50, 0.96, 0.01):
    score = compute_macro_f05(val_pairs, thresh, gt_set, all_s1_val)
    if score > best_f05:
        best_f05, best_thresh = score, thresh

print(f"Best threshold: {best_thresh:.2f} | Validation F_0.5: {best_f05:.4f}")
```

---

## Step 8: Output Generation, Validation & Export

```python
# Run model on all test candidate pairs
X_test = test_pairs[FEATURES].fillna(0).astype("float32").values
test_pairs["prob"] = model.predict_proba(X_test)[:, 1]

# Apply optimal threshold
matched = (test_pairs[test_pairs["prob"] >= best_thresh]
           .groupby("entity_id_s1")["entity_id_s23"]
           .apply(lambda x: ",".join(x.unique()))
           .reset_index()
           .rename(columns={"entity_id_s1":"source1_entity_id","entity_id_s23":"matched_entity_ids"}))

# Guarantee every S1 test entity appears — left join
all_s1_test = df_s1_test.select(pl.col("entity_id").alias("source1_entity_id")).to_pandas()
output_match = all_s1_test.merge(matched, on="source1_entity_id", how="left")
output_match["matched_entity_ids"] = output_match["matched_entity_ids"].fillna("")

output_match.to_csv("output/matching_results.tsv", sep="\t", index=False)
print(f"matching_results.tsv written | Singletons: {(output_match['matched_entity_ids']=='').sum():,}")

# Validate
import subprocess
result = subprocess.run([
    "python3", "utils/validate_submission.py",
    "--matching", "output/matching_results.tsv",
    "--candidate", "output/candidate_pairs.tsv",
    "--test-dir", "dataset/test"
], capture_output=True, text=True)
print(result.stdout)
assert result.returncode == 0, "Validation FAILED. Fix issues before submitting."
```

---

## Pipeline Summary

| Step | What | Runtime |
|---|---|---|
| 1. Ingestion | Polars TSV read, schema align, null fill | 2 min |
| 2. Normalization | FlashText, token sort, phonetic (parallel), postal extract, collision guard | 20 min |
| 3. Index Build | Hash (Polars), FAISS IVF-PQ, Inv Index (parallel), MinHash, Sorted Array → all to disk | 50 min |
| 4. Blocking | 5 blockers → union → country guard → cap → recall audit → candidate_pairs.tsv | 30 min |
| 5. Features | 55 features — rapidfuzz.cdist(workers=-1), sparse TF-IDF, embedding cosine | 60 min |
| 6. LightGBM | Entity-level split, scale_pos_weight, early stopping | 30 min |
| 7. Threshold | Sweep 0.50→0.95, optimize macro F_0.5 per S1 entity on validation | 10 min |
| 8. Export | matching_results.tsv + validator PASS | 5 min |
| **Total** | | **~3.5 hours** |

## Requirements

```
polars>=0.20
rapidfuzz>=3.0
sentence-transformers>=2.2
faiss-cpu>=1.7
datasketch>=1.6
lightgbm>=4.0
scikit-learn>=1.3
jellyfish>=1.0
flashtext>=2.7
numpy>=1.24
scipy>=1.11
pandas>=2.0
```
