"""
data_loader.py
----------------
Phase 1 - Step 1: Data Collection and Data Loading

This module is responsible for loading the arXiv research paper dataset
from a CSV file and standardizing its column names so the rest of the
app (EDA, ETA, preprocessing, TF-IDF, summarization) can rely on a
consistent schema:

    - title      : paper title (string)
    - abstract   : paper abstract (string)
    - category   : subject category / label (string)

The loader is intentionally flexible about the incoming column names
because different downloads of the "arXiv Scientific Research Paper
Dataset" use slightly different headers (e.g. "categories" vs
"category", "summary" vs "abstract").
"""

import pandas as pd
import streamlit as st

# Possible column name variants seen in different versions of arXiv datasets
TITLE_CANDIDATES = ["title", "paper_title", "Title"]
ABSTRACT_CANDIDATES = ["abstract", "summary", "Abstract", "Summary", "text"]
CATEGORY_CANDIDATES = ["category", "categories", "primary_category", "Category", "subject"]


def _find_column(df: pd.DataFrame, candidates: list[str]):
    """Return the first matching column name from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


@st.cache_data(show_spinner=False)
def load_dataset(file_path_or_buffer) -> pd.DataFrame:
    """
    Load the raw dataset from a CSV path or an uploaded file buffer,
    then standardize column names to: title, abstract, category.

    Rows that end up with a completely empty title AND abstract are
    dropped, since they carry no usable text for NLP tasks.
    """
    df = pd.read_csv(file_path_or_buffer)

    title_col = _find_column(df, TITLE_CANDIDATES)
    abstract_col = _find_column(df, ABSTRACT_CANDIDATES)
    category_col = _find_column(df, CATEGORY_CANDIDATES)

    rename_map = {}
    if title_col:
        rename_map[title_col] = "title"
    if abstract_col:
        rename_map[abstract_col] = "abstract"
    if category_col:
        rename_map[category_col] = "category"

    df = df.rename(columns=rename_map)

    # Ensure the three expected columns always exist
    for required_col in ["title", "abstract", "category"]:
        if required_col not in df.columns:
            df[required_col] = None

    # If "category" contains list-like strings such as "['cs.AI', 'cs.LG']",
    # keep only the first category for simplicity in this beginner project.
    def clean_category(val):
        if pd.isna(val):
            return val
        val = str(val)
        val = val.strip("[]").replace("'", "").replace('"', "")
        return val.split(",")[0].split()[0].strip()

    df["category"] = df["category"].apply(clean_category)

    # Drop rows with no usable text at all
    df = df.dropna(subset=["title", "abstract"], how="all").reset_index(drop=True)

    return df


def get_full_text_series(df: pd.DataFrame) -> pd.Series:
    """
    Combine title + abstract into a single text field used throughout
    the NLP pipeline (preprocessing, TF-IDF, summarization).
    """
    title = df["title"].fillna("")
    abstract = df["abstract"].fillna("")
    return (title + ". " + abstract).str.strip()
