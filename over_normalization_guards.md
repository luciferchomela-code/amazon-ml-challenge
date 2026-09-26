# Guardrails Against Over-Normalization

You have identified one of the most dangerous pitfalls in Entity Resolution. Your instinct is absolutely correct: if you normalize too aggressively (e.g., stripping out all generic words like "Corporation", "Farms", "Enterprises"), you risk **over-normalization**. 

If "Apple Corporation" and "Apple Farms" both normalize down to just "Apple", your model will incorrectly merge two different businesses. Because the evaluation metric is **F_0.5** (which heavily punishes false merges), over-normalization will destroy your score.

Here is how we can use your idea of checking against Source 1 (S1) to optimally prevent this.

---

## 1. The "S1 Collision" Test (The Ultimate Over-Normalization Check)

### The Concept
The problem statement explicitly notes that **Source 1 is the deduplicated reference source**. This is a massive hint. It means every single row in S1 represents a *unique, distinct* real-world business. 

Therefore, if you apply a normalization function to S1, and two different S1 rows suddenly end up with the *exact same* normalized name, your function is too aggressive. It destroyed the variance that made them unique.

### Optimal Implementation (Vectorized Collision Detection)
We can test our normalization rules in milliseconds using Pandas `groupby()`.

```python
import pandas as pd

def check_over_normalization(df_s1, normalized_col):
    # Group S1 by the newly normalized name and count how many original IDs share it
    collisions = df_s1.groupby(normalized_col)['entity_id'].count()
    
    # Filter for names that now map to multiple distinct S1 entities
    over_normalized = collisions[collisions > 1]
    
    if not over_normalized.empty:
        print(f"WARNING: {len(over_normalized)} S1 entities collided after normalization!")
        print(over_normalized.head())
    else:
        print("SAFE: No S1 collisions detected.")

# Run this test every time you write a new cleaning heuristic!
```

---

## 2. S1 Vocabulary Anchoring (Smart Deletion)

### The Concept
If you are doing Corpus-Aware Stopword Removal (e.g., dropping the top 1% most frequent words like "services", "ltd"), you might accidentally drop a word that is actually the *core name* of a business. (e.g., A business literally named "Services Ltd").

### Optimal Implementation ($O(1)$ Hash Sets)
Instead of blindly removing stopwords, we can build a "Golden Vocabulary" from S1. If a word is the *only* remaining word in an S1 entity's name, we protect it.

```python
# Create an O(1) lookup set of protected words (e.g., words that appear by themselves in S1)
single_word_names = df_s1[df_s1['clean_name'].str.split().str.len() == 1]['clean_name']
protected_vocab = set(single_word_names)

def safe_stopword_removal(text, stopwords, protected_vocab):
    tokens = text.split()
    # Only remove the word if it is in stopwords AND NOT in our protected S1 vocabulary
    safe_tokens = [word for word in tokens if word not in stopwords or word in protected_vocab]
    
    # Fallback: if we accidentally stripped everything, return the original text
    if not safe_tokens:
        return text
        
    return " ".join(safe_tokens)
```

---

## 3. Length & Token Guardrails

### The Concept
A simple, highly effective rule: **Never let normalization destroy a string completely.** If a normalization step results in a string that is only 1 or 2 characters long, or an empty string, the normalization should be aborted for that specific row.

### Optimal Implementation (Numpy `where`)
Using `numpy.where` is the fastest way to apply conditional logic across hundreds of thousands of rows.

```python
import numpy as np

# Apply your aggressive normalization
df_all['aggressive_name'] = df_all['clean_name'].apply(my_aggressive_function)

# Guardrail: If the resulting string is less than 3 characters, 
# revert back to the basic 'clean_name'.
df_all['final_name'] = np.where(
    df_all['aggressive_name'].str.len() < 3, 
    df_all['clean_name'],            # Condition False: Revert to safe version
    df_all['aggressive_name']        # Condition True: Keep aggressive version
)
```
