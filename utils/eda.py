"""
eda.py
-------
Phase 1 - Step 2: Exploratory Data Analysis (EDA)

EDA here focuses on the STRUCTURAL / STATISTICAL properties of the
dataset as a table - shape, missing values, duplicates, category
distribution, text length, and word frequency. This is the classic
"get to know your dataframe" analysis that applies to almost any
tabular dataset, text or not.

(Contrast this with eta.py, which performs NLP-specific *linguistic*
analysis - see that file's docstring for the distinction.)
"""

import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def get_shape_and_columns(df: pd.DataFrame) -> dict:
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def get_missing_and_duplicates(df: pd.DataFrame) -> dict:
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_table = pd.DataFrame({
        "missing_count": missing,
        "missing_percent": missing_pct,
    })
    duplicate_rows = int(df.duplicated().sum())
    return {
        "missing_table": missing_table,
        "duplicate_rows": duplicate_rows,
    }


def get_category_distribution(df: pd.DataFrame) -> pd.Series:
    return df["category"].value_counts(dropna=False)


def plot_category_distribution(category_counts: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 5))
    category_counts.sort_values(ascending=True).plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Number of Papers")
    ax.set_ylabel("Category")
    ax.set_title("Category Distribution")
    fig.tight_layout()
    return fig


def compute_text_lengths(df: pd.DataFrame) -> pd.DataFrame:
    lengths = pd.DataFrame()
    lengths["title_char_len"] = df["title"].fillna("").apply(len)
    lengths["title_word_count"] = df["title"].fillna("").apply(lambda x: len(x.split()))
    lengths["abstract_char_len"] = df["abstract"].fillna("").apply(len)
    lengths["abstract_word_count"] = df["abstract"].fillna("").apply(lambda x: len(x.split()))
    return lengths


def plot_text_length_distribution(lengths: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(lengths["abstract_word_count"], bins=30, ax=axes[0], color="#55A868")
    axes[0].set_title("Abstract Length (word count)")
    axes[0].set_xlabel("Words")

    sns.histplot(lengths["title_word_count"], bins=20, ax=axes[1], color="#C44E52")
    axes[1].set_title("Title Length (word count)")
    axes[1].set_xlabel("Words")
    fig.tight_layout()
    return fig


def get_top_frequent_words(text_series: pd.Series, top_n: int = 20) -> pd.DataFrame:
    """
    Simple raw-text word frequency count (BEFORE any NLP preprocessing).
    This is a quick, surface-level EDA check - it will include common
    stopwords like 'the' and 'of', which is expected at this stage.
    """
    all_text = " ".join(text_series.fillna("").astype(str).tolist()).lower()
    words = re.findall(r"[a-zA-Z]+", all_text)
    counts = Counter(words)
    top_words = counts.most_common(top_n)
    return pd.DataFrame(top_words, columns=["word", "frequency"])


def plot_top_words(top_words_df: pd.DataFrame, title: str = "Top Frequent Words"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=top_words_df.sort_values("frequency"), x="frequency", y="word", ax=ax, color="#8172B2")
    ax.set_title(title)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Word")
    fig.tight_layout()
    return fig
