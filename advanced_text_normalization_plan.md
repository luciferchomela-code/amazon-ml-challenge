# Advanced & Clever Text Normalization Strategies

You are absolutely right. In Entity Resolution, a model is only as good as the data you feed it. If you rely solely on heavy Machine Learning algorithms to figure out messy strings, the model will struggle. **Clever normalization solves 80% of the problem before ML even starts.**

Here is a much deeper, more advanced approach to text normalization that uses "clever" heuristics specifically designed for business names and addresses.

---

## 1. Phonetic Hashing (Catching "Sounds-Like" Typos)

### The Problem
Typos aren't random; they are often phonetic. This is especially true for transliterated data (e.g., from India). "Bazaar" vs "Bazar", "Kapur" vs "Kapoor", or "Smyth" vs "Smith". Standard string distances (like Levenshtein) penalize these heavily.

### The Clever Approach
Do not just clean the text; **translate the text into how it sounds**. Use algorithms like **Double Metaphone** or **Soundex**. These algorithms convert English (and transliterated) words into phonetic codes.
*   `Bazaar` $\rightarrow$ `BSR`
*   `Bazar` $\rightarrow$ `BSR`

### Implementation
Store this as a completely separate column. When the ML model compares S1 and S2, it won't just look at how similar the letters are; it will look at how similar the *phonetic codes* are.
```python
# pip install jellyfish
import jellyfish

def get_phonetic_code(text):
    if not text: return ""
    # Double Metaphone handles foreign words/names better than Soundex
    return jellyfish.metaphone(text)

df_all['phonetic_name'] = df_all['clean_name'].apply(get_phonetic_code)
```

---

## 2. Token Sorting (Defeating Word-Order Transpositions)

### The Problem
The problem statement explicitly warns about "word-order transpositions". 
*   String A: `"SBI ATM Near Main Road"`
*   String B: `"Near Main Road SBI ATM"`
To a standard string distance algorithm, these look totally different (high edit distance).

### The Clever Approach
Create a version of the name and address where the words are alphabetically sorted. 
*   String A Sorted: `"atm main near road sbi"`
*   String B Sorted: `"atm main near road sbi"`
Suddenly, two vastly different strings become **exact identical matches**.

### Implementation
```python
def sort_tokens(text):
    # Split by space, sort alphabetically, rejoin
    return " ".join(sorted(text.split()))

df_all['sorted_name'] = df_all['clean_name'].apply(sort_tokens)
```

---

## 3. Heuristic Address Parsing (Isolating High-Signal Data)

### The Problem
An entire address string is noisy. Comparing `"123 Main St, Apt 4, NY 10001"` to `"Apt 4, 123 Main Street, New York, 10001"` is messy.

### The Clever Approach
Don't just clean the address as one big chunk. **Extract the highest-signal sub-components into their own columns using Regex.** The most powerful identifier in an address is the numeric postal code.

*   **US ZIP Codes**: 5 or 9 digits (`10001` or `10001-1234`)
*   **India PIN Codes**: 6 digits (`400001`)

### Implementation
Extract these before you strip punctuation, so you can isolate them safely.
```python
# Regex to find 5-digit US ZIPs or 6-digit India PINs
postal_code_pattern = r'\b(\d{5}(?:-\d{4})?|\d{6})\b'

# Extract into a new column
df_all['postal_code'] = df_all['business_address'].str.extract(postal_code_pattern)

# If two records have the exact same postal code, the probability 
# of them being a match skyrockets. You can use this as a massive 
# boost feature for your ML model.
```

---

## 4. Acronym Squeezing

### The Problem
The problem warns about abbreviations. "I.B.M." vs "IBM" vs "I B M". If you just strip punctuation, "I.B.M." becomes "I B M", which still doesn't match "IBM".

### The Clever Approach
Before your global clean, look for single letters separated by spaces or dots, and squeeze them together.

### Implementation
```python
import re

# Finds single letters separated by spaces (e.g. "I B M") and removes the space
def squeeze_acronyms(text):
    # Replace "I B M" with "IBM"
    squeezed = re.sub(r'(?<=\b[a-zA-Z])\s+(?=[a-zA-Z]\b)', '', text)
    return squeezed
```

---

## 5. Corpus-Aware Dynamic Stopword Removal

### The Problem
Hardcoding a list of words to remove (like "and", "the", "near") is brittle and you might miss domain-specific garbage words.

### The Clever Approach
Let the data tell you what is useless. Calculate the frequency of every word across the entire `business_name` column. The top 1% most frequent words are almost certainly generic legal terms ("corporation", "ltd", "services", "enterprises"). 
*   **Strip the generic words out, leaving only the highly unique, discriminative core identifier.** 
*   "Starbucks Corporation" $\rightarrow$ "Starbucks". 
*   Now you are only comparing the words that actually matter.
