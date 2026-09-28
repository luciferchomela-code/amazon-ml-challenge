# Amazon ML Challenge 2024 — Business Entity Resolution

A scalable machine learning pipeline for resolving business entities across multiple noisy data sources.

This project was developed for the **Amazon ML Challenge 2024**, where the task was to identify records from different sources that refer to the same real-world business.

Our solution ranked approximately **Top 2000 among 90,000+ teams**.

---

## Problem

Business information often comes from multiple independent sources. The same business may appear with:

* Different names
* Misspellings and OCR errors
* Different address formats
* Missing fields
* Abbreviations
* Transliteration differences
* Inconsistent postal codes
* Different representations of the same entity

For example:

```text
"Rddy Txtls"
"Reddy Textiles Pvt Ltd"
```

may refer to the same business.

Similarly:

```text
"5olutions Inc"
"Solutions Incorporated"
```

may represent the same entity.

With millions of records, comparing every possible pair is computationally infeasible.

For 26 million records, an exhaustive pairwise comparison would require an enormous number of comparisons, making efficient candidate generation essential.

---

## Solution Overview

Our pipeline consists of three major stages:

```text
Raw Data
   │
   ▼
Data Cleaning & Normalization
   │
   ▼
Candidate Generation / Blocking
   │
   ▼
Feature Engineering
   │
   ▼
LightGBM Classifier
   │
   ▼
Threshold Optimization
   │
   ▼
Entity Matches
```

The main goal was to aggressively reduce the number of candidate pairs while retaining high-quality potential matches.

---

## 1. Data Cleaning and Normalization

Before generating candidate pairs, textual fields were normalized to reduce superficial differences.

Typical preprocessing included:

* Lowercasing
* Removing unnecessary punctuation
* Standardizing whitespace
* Normalizing common textual patterns
* Extracting numeric components
* Processing postal/PIN codes
* Preparing names and addresses for fuzzy matching

This helped make different representations of the same entity more comparable.

---

## 2. Candidate Generation / Blocking

A naive approach would compare every record against every other record.

For large datasets, this quickly becomes infeasible.

Instead, we used a **multi-pass blocking strategy**.

Candidate pairs were generated using several signals, including:

* Phonetic representations
* Consonant signatures
* Postal/PIN code information
* Name-based blocking
* Address-based blocking

Different blocking strategies were combined to improve recall while keeping the candidate set manageable.

### Why blocking?

Suppose we have two records:

```text
Record A:
Reddy Textiles Pvt Ltd
Ranchi, 834001

Record B:
Rddy Txtls
Ranchi, 834001
```

Comparing these records is useful.

But comparing them with every business in the dataset is unnecessary.

Blocking allows us to focus computation on records that have at least some evidence of potentially belonging to the same entity.

---

## 3. Polars for Large-Scale Processing

We used **Polars** for data processing and candidate generation.

Polars was useful because of its:

* Fast execution
* Efficient memory usage
* Lazy evaluation
* Columnar processing
* Rust-based implementation

Candidate generation and feature computation were performed in batches where necessary to avoid excessive memory consumption.

---

## 4. Feature Engineering

For every candidate pair, we generated a set of similarity and structural features.

In total, our model used approximately **18 features**.

### String Similarity

We used several fuzzy matching techniques:

* RapidFuzz `partial_ratio`
* RapidFuzz `token_sort_ratio`
* RapidFuzz `token_set_ratio`
* Levenshtein similarity
* Jaro-Winkler similarity

These features capture different types of textual similarity.

For example:

```text
Reddy Textiles Pvt Ltd
Rddy Txtls
```

may have a relatively low exact-match similarity but still exhibit strong fuzzy similarity.

---

### Character N-Gram Similarity

We also used TF-IDF based character n-gram representations.

The model considered character n-grams in the range:

```text
2-gram → 4-gram
```

and calculated cosine similarity between the corresponding fields.

Character n-grams are particularly useful for handling:

* Typos
* Missing characters
* Abbreviations
* OCR errors
* Minor spelling variations

---

### Numeric Address Features

Addresses often contain useful structured information.

We extracted and compared numeric components such as:

* PIN/postal codes
* Building numbers
* Other numeric address components

These features provided additional evidence beyond raw text similarity.

---

### Phonetic Features

Phonetic representations were used to capture names that sound similar despite having different spellings.

This can help with variations caused by:

* Transliteration
* Spelling errors
* Regional naming variations

---

### Length-Based Features

We also included structural features such as:

* Name length difference
* Address length difference
* Relative length differences

These features provide additional context to the classifier.

---

## 5. LightGBM Classifier

After candidate generation and feature extraction, the candidate pairs were passed to a **LightGBM binary classifier**.

The classifier predicts whether two records represent the same real-world entity.

Conceptually:

```text
Candidate Pair
      │
      ▼
Feature Extraction
      │
      ▼
18 Features
      │
      ▼
LightGBM
      │
      ▼
Match Probability
```

LightGBM was chosen because it performs well on tabular feature sets and is computationally efficient.

---

## 6. Threshold Optimization

One of the most important parts of the solution was optimizing the classification threshold according to the actual competition metric.

The competition evaluated performance using **F0.5**, which gives more importance to precision than recall.

The formula is:

```text
F0.5 = (1 + 0.5²) × Precision × Recall
       ----------------------------------
       (0.5² × Precision) + Recall
```

Instead of simply using:

```text
probability >= 0.5
```

we performed a threshold search to find a value better suited to the evaluation metric.

We evaluated thresholds using a validation pool that included both candidate matches and zero-candidate cases.

This was important because entity resolution has an asymmetric cost:

```text
False Positive
     ↓
Incorrectly merging two different businesses

False Negative
     ↓
Failing to link two records belonging to the same business
```

For this competition, precision was particularly important.

---

## 7. Memory-Efficient Processing

The dataset was large enough that processing everything at once could cause excessive memory usage.

To address this, we used:

* Chunked data processing
* Batch feature computation
* Efficient Polars operations
* Incremental model training where applicable
* Avoiding unnecessary copies of large datasets

The goal was to make the entire pipeline executable without requiring a GPU or large distributed cluster.

---

## Results

Our final solution achieved approximately:

| Metric      |                   Result |
| ----------- | -----------------------: |
| Competition | Amazon ML Challenge 2024 |
| Teams       |                  90,000+ |
| Rank        |                    ~2000 |
| F0.5        |                     0.7+ |
| GPU         |             Not required |

The exact score and ranking may vary depending on the final competition evaluation and leaderboard snapshot.

---

## Tech Stack

### Programming

* Python

### Data Processing

* Polars
* Pandas

### Machine Learning

* LightGBM
* Scikit-learn

### NLP / Similarity

* RapidFuzz
* TF-IDF
* Character n-grams
* Jaro-Winkler
* Levenshtein
* Phonetic matching

---

## Project Structure

```text
amazon-ml-challenge/
│
├── data/
│   └── README.md
│
├── preprocessing/
│   └── ...
│
├── blocking/
│   └── ...
│
├── features/
│   └── ...
│
├── model/
│   └── ...
│
├── inference/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── requirements.txt
├── README.md
└── ...
```

The exact structure may vary depending on the version of the repository.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/luciferchomela-code/amazon-ml-challenge.git

cd amazon-ml-challenge
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

The overall workflow is:

```text
1. Prepare the input data
2. Normalize the records
3. Generate candidate pairs
4. Generate pairwise features
5. Train the LightGBM model
6. Optimize the F0.5 threshold
7. Run inference
8. Generate the final predictions
```

Refer to the individual scripts/notebooks in the repository for the corresponding commands.

---

## Key Takeaways

The biggest lesson from this challenge was that **model selection was only one part of the problem**.

For large-scale entity resolution, the overall system depends heavily on:

1. Efficient candidate generation
2. High-quality feature engineering
3. Memory-efficient processing
4. Correct evaluation methodology
5. Threshold optimization

A strong classifier cannot compensate for poor candidate generation, and optimizing the wrong evaluation metric can lead to suboptimal predictions.

In our case, switching our optimization target from F1 to the competition's F0.5 metric resulted in a significant improvement.

---

## Team

This project was developed as a team of four for the Amazon ML Challenge 2024.

---

## License

This repository is intended for educational and research purposes.

Please refer to the repository and competition rules for any restrictions regarding the original competition data.

---

## Repository

GitHub:

https://github.com/luciferchomela-code/amazon-ml-challenge
