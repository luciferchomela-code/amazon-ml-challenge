"""
inspect_false_positives.py
Read-only diagnostic. Does NOT modify step4/step5. Reuses their exact
logic/imports to reproduce the same validation split and threshold, then
prints which pairs the model incorrectly predicted as matches, so we can
see if Postal_Only is responsible.
"""
import sys
from pathlib import Path

import polars as pl
from sklearn.model_selection import GroupShuffleSplit

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from step5_model import FEATURES, train_model

clf, threshold = train_model(
    "output/cache/features_train.parquet",
    s1_train_path="output/cache/train_source1.parquet",
    gt_path="output/cache/train_ground_truth.parquet",
)

df = pl.read_parquet("output/cache/features_train.parquet").to_pandas()
df[FEATURES] = df[FEATURES].fillna(0)

_, validation_indices = next(
    GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42).split(
        df, groups=df["entity_id_s1"]
    )
)
df = df.iloc[validation_indices].copy()
df["prob"] = clf.predict_proba(df[FEATURES])[:, 1]

false_positives = df[(df["prob"] >= threshold) & (df["label"] == 0)]
print(f"False positives at threshold {threshold:.2f}: {len(false_positives)}")
print(false_positives[["entity_id_s1", "entity_id_s23", "prob"] + FEATURES].to_string(index=False))