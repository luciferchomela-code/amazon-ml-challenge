# Methodology Write-Up: Business Entity Resolution

## 1. Methodology Used
Our methodology focuses on maximizing both computational efficiency and precision for large-scale Entity Resolution (1M+ records per source). Standard ER pipelines often fail at this scale due to OOM (Out-Of-Memory) crashes during the candidate generation phase or severe computational bottlenecks during string similarity calculations. 

To overcome this, we implemented a pure-Polars, zero-neural-embedding architecture. The pipeline operates in five tightly coupled stages:
1. **Ingestion & Caching**: Parquet-backed, BOM-safe loading that guarantees zero silent failures.
2. **Standardization**: O(N) Aho-Corasick automaton (FlashText) abbreviation expansion and Regex Acronym Squeezing.
3. **Blocking**: Multi-Pass Mega-Block Pruned Joins (an optimized variant of the Sorted Neighborhood Method).
4. **Feature Engineering**: Vectorized Polars heuristics, Multiprocessing RapidFuzz, and Scipy Sparse Matrix TF-IDF char n-gram dot products.
5. **Modeling & Thresholding**: LightGBM trained with Entity-Level splits (`GroupShuffleSplit`) to strictly prevent data leakage, paired with a custom Macro F_0.5 mathematical threshold optimizer.

## 2. Candidate Generation/Blocking Strategy
Instead of relying on computationally heavy FAISS inverted indexes or LLM embeddings—which would require hours of CPU inference—we developed a **Multi-Pass Mega-Block Pruned Join** strategy.

We executed 6 separate exact-match fuzzy joins in Polars (which run natively in C++ in milliseconds):
1. **Exact Name**: `clean_name`
2. **Sorted Name**: `sorted_name` (Tokens alphabetized to catch word-order inversions).
3. **Phonetic Hash**: Double Metaphone of the first name token.
4. **Name 1st Word + Postal**: Captures branch-level granularity.
5. **Name (10 chars) + Address (10 chars)**: Captures highly specific sub-strings.
6. **Address Numerics + Name 1st Word**: Extracts digits from messy addresses paired with the core business name.

**Cartesian Explosion Guard (Mega-Block Pruning):**
The primary risk of blocking is joining on a generic key (e.g., "McDonalds" occurring 10,000 times), creating a massive Cartesian product that crashes RAM. Our script automatically groups and counts the occurrences of every block key in the S2/S3 candidate pool. If a key's frequency exceeds a predefined `max_size` (e.g., 5000), it is classified as a "mega-block" and aggressively pruned. The loss in recall is naturally mitigated by the other 5 overlapping blocking strategies.

## 3. Model Architecture and Feature Engineering

### Feature Engineering
We engineered 13 highly orthogonal features without resorting to slow $O(N \times M)$ pairwise string loops:
*   **TF-IDF Sparse Dot Products**: Instead of running `.transform()` on 25M pairs, we fit a `TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))` on unique entities, mapped the array indices, and used a hyper-fast Scipy row-wise sparse dot product (`va.multiply(vb).sum(axis=1)`). This converts NLP string distances into a 5-second mathematical operation. Character word-boundary (`char_wb`) n-grams were specifically chosen as they mathematically excel at detecting transliteration typos (e.g., "Kapur" vs "Kapoor").
*   **Bypassing the Python GIL**: `rapidfuzz` string metrics (Jaro-Winkler, Token Sort, Token Set, Levenshtein) were chunked and passed to a `multiprocessing.Pool`, distributing the C-backend execution across all CPU cores.
*   **Polars Native Heuristics**: Computed Length Ratios, First-Word matches, and Country/Source guards natively in SIMD.

### Model Architecture
We utilized **LightGBM**, configured with `scale_pos_weight` to mathematically force the trees to pay attention to the severe class imbalance (where true matches represent <1% of the candidate pairs). 

**Data Leakage Prevention:**
A common error in ER is performing a standard `train_test_split` on candidate pairs. If candidate pairs belonging to the same Source 1 Entity are split across Train and Validation, the model overfits to the entity's signature. We enforced `GroupShuffleSplit(groups=entity_id_s1)` to guarantee 100% isolation of entities between splits.

## 4. Any other relevant information about the approach
**Custom Macro F_0.5 Threshold Sweeper**
Standard ML models output probabilities and default to a `0.50` decision threshold. However, the competition heavily biases towards Precision ($F_{0.5}$). 

To optimize this, our pipeline ignores the default threshold. It instead replicates the exact mathematical Leaderboard scoring formula locally. During the validation phase, the pipeline dynamically sweeps probabilities from `0.40` to `0.98`, calculating the true macro-averaged $F_{0.5}$ (including the 1.0/0.0 scoring logic for S1 singletons). It mathematically extracts the exact threshold that maximizes the leaderboard score, which is then exclusively used for test-set inference.
