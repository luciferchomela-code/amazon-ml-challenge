# Implementation Plan: Data Ingestion & Schema Alignment

This document outlines the optimal, highly performant technical implementation for the first phase of the Entity Resolution pipeline. The primary focus is memory efficiency and vectorized operations.

## Technology Stack Recommendation
While `pandas` is the standard, for the **most optimal** processing of large-scale commercial datasets, consider using **Polars** (`import polars as pl`). Polars is written in Rust, natively multi-threaded, and significantly faster and more memory-efficient than Pandas for parsing and string manipulation. 
*Note: The implementation details below use Pandas syntax as it was referenced in your problem statement, but the principles apply equally to Polars.*

---

## 1. Secure Parsing

### Objective
Safely import the `.tsv` files without corrupting data that contains commas, while minimizing the memory footprint during the I/O operation.

### Optimal Implementation Details
1. **Explicit Delimiter & Quoting**: 
   - Always use `sep="\t"`.
   - Set `quoting=csv.QUOTE_NONE` if the TSV is strictly tab-separated and does not use quotes to encapsulate tabs. This prevents the parser from getting confused by stray quotation marks inside business names.
2. **Schema-on-Read (Pre-defining Dtypes)**:
   - Do not let the library infer data types. Strings loaded as generic `object` types consume massive memory.
   - Force all columns to the modern `string` dtype (or PyArrow strings `string[pyarrow]`) during the read operation.
3. **Handling Bad Lines**:
   - Set `on_bad_lines='error'` during development to catch malformed rows immediately, rather than silently dropping them.

**Code Example (Pandas):**
```python
import pandas as pd
import csv

# Define highly optimized dtypes
dtypes = {
    'entity_id': 'string[pyarrow]',
    'business_name': 'string[pyarrow]',
    'business_address': 'string[pyarrow]',
    'country': 'category' # Categorical is highly optimized for low-cardinality sets
}

# Read securely
df_s1 = pd.read_csv("dataset/train/train_source1.tsv", 
                    sep="\t", 
                    dtype=dtypes, 
                    quoting=csv.QUOTE_NONE)
```

---

## 2. Schema Mapping & Standardization

### Objective
Unify the data structures so downstream modules (cleaning, blocking) only have to be written once and applied to a single dataset, while handling missing values safely.

### Optimal Implementation Details
1. **Unified DataFrame via Concatenation**:
   - Instead of maintaining `df_s1`, `df_s2`, and `df_s3`, concatenate them vertically into one master DataFrame immediately after reading. Vectorized text cleaning on 300k rows once is faster than 100k rows three times.
2. **Missing Value Imputation**:
   - Text distance algorithms (like Levenshtein) will crash if they receive a `NaN` or `None` float type instead of a string.
   - Vectorized replace: Immediately fill all `NaN`/`Null` values in text columns with empty strings `""`.

**Code Example (Pandas):**
```python
# Combine sources
df_all = pd.concat([df_s1, df_s2, df_s3], ignore_index=True)

# Safe imputation for text features
cols_to_fill = ['business_name', 'business_address']
df_all[cols_to_fill] = df_all[cols_to_fill].fillna("")
```

---

## 3. Source Tagging

### Objective
Programmatically isolate the source origin (S1, S2, S3) using the `entity_id` prefixes to route data correctly later (e.g., ensuring we only generate blocks between S1 and S2/S3, never S1 vs S1).

### Optimal Implementation Details
1. **Vectorized String Slicing**:
   - **Anti-Pattern**: Never use `.apply(lambda x: x.split('-')[0])`. Iterating row-by-row in Python is incredibly slow.
   - **Optimal Pattern**: Use the library's built-in vectorized C-level string accessors (`.str`). Since the prefixes are exactly two characters ("S1", "S2", "S3"), string slicing is faster than splitting by a delimiter.
2. **Type Casting to Categorical**:
   - Once extracted, the source tag only has 3 possible values. Cast this new column to a `category` dtype to save significant memory and speed up future boolean filtering.

**Code Example (Pandas):**
```python
# Extract the first two characters (S1, S2, S3) optimally
df_all['source_tag'] = df_all['entity_id'].str[:2]

# Convert to memory-efficient categorical dtype
df_all['source_tag'] = df_all['source_tag'].astype('category')

# Now you can easily filter later:
# source_1_records = df_all[df_all['source_tag'] == 'S1']
```
