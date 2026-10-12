"""
preprocessing.py
----------------
Phase 1 - Step 4: Data Preprocessing
NLP Concept 1: Text Preprocessing

Pipeline:
1. Lowercasing
2. Punctuation removal
3. Tokenization
4. Stopword removal
5. Lemmatization

This version uses safe NLTK handling so that the Streamlit
application does not crash when an NLTK resource is unavailable.
"""

import re
import string

import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# ---------------------------------------------------------
# NLTK RESOURCE SETUP
# ---------------------------------------------------------

def get_stopwords():
    """Load English stopwords safely."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))
        except Exception:
            # Basic fallback stopwords
            return {
                "a", "an", "the", "and", "or", "but", "if",
                "is", "are", "was", "were", "to", "of", "in",
                "on", "for", "with", "as", "by", "at", "from",
                "this", "that", "these", "those", "it", "its",
                "be", "been", "being", "we", "our", "they",
                "their", "can", "could", "would", "should",
                "will", "also", "using", "used"
            }


STOPWORDS = get_stopwords()


def get_lemmatizer():
    """Create WordNet lemmatizer safely."""
    try:
        nltk.data.find("corpora/wordnet")
        return WordNetLemmatizer()
    except LookupError:
        try:
            nltk.download("wordnet", quiet=True)
            return WordNetLemmatizer()
        except Exception:
            return None


LEMMATIZER = get_lemmatizer()


# ---------------------------------------------------------
# STEP 1: LOWERCASE
# ---------------------------------------------------------

def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    if not isinstance(text, str):
        return ""

    return text.lower()


# ---------------------------------------------------------
# STEP 2: REMOVE PUNCTUATION
# ---------------------------------------------------------

def remove_punctuation(text: str) -> str:
    """Remove punctuation and unwanted symbols."""
    if not isinstance(text, str):
        return ""

    # Remove standard punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove remaining non-alphanumeric characters
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ---------------------------------------------------------
# STEP 3: TOKENIZATION
# ---------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Split text into individual words safely."""
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        return word_tokenize(text)

    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
            return word_tokenize(text)

        except Exception:
            # Simple fallback tokenizer
            return text.split()


# ---------------------------------------------------------
# STEP 4: STOPWORD REMOVAL
# ---------------------------------------------------------

def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove common English stopwords."""
    return [
        token
        for token in tokens
        if token.lower() not in STOPWORDS
        and len(token) > 1
    ]


# ---------------------------------------------------------
# STEP 5: LEMMATIZATION
# ---------------------------------------------------------

def lemmatize(tokens: list[str]) -> list[str]:
    """Convert words to their base form."""
    if not tokens:
        return []

    # If WordNet is unavailable, return tokens unchanged
    if LEMMATIZER is None:
        return tokens

    result = []

    for token in tokens:
        try:
            result.append(
                LEMMATIZER.lemmatize(token)
            )
        except Exception:
            result.append(token)

    return result


# ---------------------------------------------------------
# COMPLETE PREPROCESSING PIPELINE
# ---------------------------------------------------------

def preprocess_text(text: str) -> dict:
    """
    Run the complete preprocessing pipeline on one document.

    Returns:
        original
        lowercased
        no_punctuation
        tokens
        tokens_no_stopwords
        lemmatized_tokens
        cleaned_text
    """

    if not isinstance(text, str) or not text.strip():
        return {
            "original": text if text is not None else "",
            "lowercased": "",
            "no_punctuation": "",
            "tokens": [],
            "tokens_no_stopwords": [],
            "lemmatized_tokens": [],
            "cleaned_text": "",
        }

    # Step 1
    lowercased = to_lowercase(text)

    # Step 2
    no_punct = remove_punctuation(lowercased)

    # Step 3
    tokens = tokenize(no_punct)

    # Step 4
    tokens_no_sw = remove_stopwords(tokens)

    # Step 5
    lemmatized = lemmatize(tokens_no_sw)

    # Final cleaned text
    cleaned_text = " ".join(lemmatized)

    return {
        "original": text,
        "lowercased": lowercased,
        "no_punctuation": no_punct,
        "tokens": tokens,
        "tokens_no_stopwords": tokens_no_sw,
        "lemmatized_tokens": lemmatized,
        "cleaned_text": cleaned_text,
    }


# ---------------------------------------------------------
# PROCESS COMPLETE DATASET
# ---------------------------------------------------------

def preprocess_series(text_series: pd.Series) -> pd.Series:
    """
    Apply preprocessing to every document in a pandas Series.

    Returns only the final cleaned text.
    """

    return text_series.fillna("").astype(str).apply(
        lambda text: preprocess_text(text)["cleaned_text"]
    )


# ---------------------------------------------------------
# BEFORE / AFTER EVALUATION
# ---------------------------------------------------------

def compute_before_after_stats(
    original_series: pd.Series,
    cleaned_series: pd.Series
) -> pd.DataFrame:
    """
    Compare text statistics before and after preprocessing.

    Metrics:
    - Average word count
    - Vocabulary size
    - Total word count
    - Reduction percentage
    """

    original_series = (
        original_series
        .fillna("")
        .astype(str)
    )

    cleaned_series = (
        cleaned_series
        .fillna("")
        .astype(str)
    )

    # Word counts
    orig_word_counts = original_series.apply(
        lambda text: len(text.split())
    )

    clean_word_counts = cleaned_series.apply(
        lambda text: len(text.split())
    )

    # Vocabulary
    original_words = []

    for text in original_series:
        original_words.extend(
            text.lower().split()
        )

    cleaned_words = []

    for text in cleaned_series:
        cleaned_words.extend(
            text.lower().split()
        )

    orig_vocab = set(original_words)
    clean_vocab = set(cleaned_words)

    # Statistics
    stats = pd.DataFrame({
        "Metric": [
            "Average word count per document",
            "Total vocabulary size",
            "Total word count"
        ],

        "Before Preprocessing": [
            round(orig_word_counts.mean(), 2),
            len(orig_vocab),
            int(orig_word_counts.sum())
        ],

        "After Preprocessing": [
            round(clean_word_counts.mean(), 2),
            len(clean_vocab),
            int(clean_word_counts.sum())
        ]
    })

    # Reduction percentage
    reductions = []

    for before, after in zip(
        stats["Before Preprocessing"],
        stats["After Preprocessing"]
    ):
        if before == 0:
            reductions.append(0.0)
        else:
            reduction = (
                (before - after)
                / before
                * 100
            )

            reductions.append(
                round(reduction, 2)
            )

    stats["Reduction %"] = reductions

    return stats