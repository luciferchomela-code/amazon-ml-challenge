# Entity Resolution Pipeline Architecture

## 1. Data Ingestion & Schema Alignment
*   **Secure Parsing:** Reading `.tsv` files with explicit tab delimiters to protect internal commas in address and ID fields.
*   **Schema Mapping:** Standardizing the dataset structure across all three sources, ensuring uniform column types and handling missing values (NaN/Null) consistently.
*   **Source Tagging:** Programmatically isolating the source origin (S1, S2, S3) using the `entity_id` prefixes to route data correctly in downstream tasks.

## 2. Text Normalization & Standardization
*   **Global Cleansing:** Converting all text to lowercase, removing non-alphanumeric characters, and collapsing whitespace.
*   **Domain-Specific Standardization:** Utilizing lookup dictionaries to normalize business legal suffixes (e.g., *Corp* $\rightarrow$ *Corporation*) and address components (e.g., *St* $\rightarrow$ *Street*).
*   **Open-Set Categorical Handling:** Configuring the pipeline to accept dynamic country labels dynamically (e.g., France) without hardcoded filtering.

## 3. Indexing & Blocking (Candidate Generation)
*   **Key Generation:** Creating blocking keys (e.g., exact Country + first 4 characters of the business name) to partition the data.
*   **Candidate Pair Formulation:** Generating pairwise combinations strictly between Source 1 and Source 2/Source 3 entities that share the same blocking keys.
*   **Candidate Export:** Finalizing the candidate search space and outputting the exact `candidate_pairs.tsv` file required for methodology evaluation.

## 4. Pairwise Feature Engineering
*   **Lexical Similarity Computation:** Calculating string-level edit distances (e.g., Levenshtein, Jaro-Winkler) for `business_name` and `business_address`.
*   **Vector/Token Similarity Computation:** Calculating set-based similarities (e.g., Jaccard index, TF-IDF Cosine Similarity) to handle word reordering and missing tokens.
*   **Feature Concatenation:** Assembling the computed similarity metrics into a structured numerical feature vector for every candidate pair.

## 5. Model Inference & Scoring
*   **Predictive Inference:** Passing the numerical feature vectors through the trained Machine Learning classifier.
*   **Probability Scoring:** Outputting a continuous probability score (0.0 to 1.0) indicating the model's confidence that the candidate pair represents the same real-world entity.

## 6. Decision Thresholding & Resolution
*   **Precision-Weighted Thresholding:** Applying a strict cut-off threshold to the probability scores, deliberately tuned high to optimize for the F_0.5 metric (minimizing false positives).
*   **Singleton Identification:** Explicitly flagging Source 1 entities where no candidate pairs surpassed the confidence threshold.

## 7. Output Generation & Validation
*   **List Aggregation:** Grouping all successfully matched Source 2 and Source 3 IDs for each Source 1 entity into comma-separated strings.
*   **Constraint Enforcement:** Ensuring empty strings are assigned to Singletons and verifying no duplicate IDs exist within the aggregated lists.
*   **Final Export:** Generating the final `matching_results.tsv` file and passing it through the `utils/validate_submission.py` script.
