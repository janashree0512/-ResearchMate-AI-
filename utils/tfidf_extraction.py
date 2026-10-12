"""
tfidf_extraction.py
--------------------
NLP Concept 2: TF-IDF Feature Extraction

This module:
1. Fits a TF-IDF vectorizer on cleaned research abstracts.
2. Finds important words across the complete dataset.
3. Finds important words for an individual paper.
4. Evaluates the TF-IDF feature matrix.
5. Creates a stable Matplotlib chart for the Streamlit dashboard.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------
# CHART STYLE
# ---------------------------------------------------------

sns.set_style("whitegrid")


# ---------------------------------------------------------
# TF-IDF FITTING
# ---------------------------------------------------------

def fit_tfidf(
    cleaned_text_series: pd.Series,
    max_features: int = 2000
):
    """
    Fit a TF-IDF vectorizer on cleaned text.

    Parameters:
        cleaned_text_series : cleaned document text
        max_features       : maximum vocabulary size

    Returns:
        vectorizer
        tfidf_matrix
        feature_names
    """

    # Remove missing values
    text_data = (
        cleaned_text_series
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Remove completely empty documents
    text_data = text_data[text_data != ""]

    # Safety check
    if text_data.empty:
        return None, None, np.array([])

    vectorizer = TfidfVectorizer(
        max_features=max_features
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(text_data)
    except ValueError:
        # Happens when there are no usable words
        return None, None, np.array([])

    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    return (
        vectorizer,
        tfidf_matrix,
        feature_names
    )


# ---------------------------------------------------------
# TOP WORDS - OVERALL CORPUS
# ---------------------------------------------------------

def get_top_words_overall(
    tfidf_matrix,
    feature_names: np.ndarray,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Find the most important words across the complete
    research-paper corpus using average TF-IDF score.
    """

    # Safety check
    if (
        tfidf_matrix is None
        or feature_names is None
        or len(feature_names) == 0
    ):
        return pd.DataFrame(
            columns=[
                "word",
                "avg_tfidf_score"
            ]
        )

    # Calculate average TF-IDF
    avg_scores = np.asarray(
        tfidf_matrix.mean(axis=0)
    ).flatten()

    # Replace invalid values
    avg_scores = np.nan_to_num(
        avg_scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Get top words
    top_indices = (
        avg_scores
        .argsort()[::-1][:top_n]
    )

    result = pd.DataFrame({
        "word": feature_names[top_indices],
        "avg_tfidf_score": np.round(
            avg_scores[top_indices],
            4
        )
    })

    # Final safety cleaning
    result = result.replace(
        [np.inf, -np.inf],
        0
    )

    result = result.dropna(
        subset=["word", "avg_tfidf_score"]
    )

    return result


# ---------------------------------------------------------
# TOP WORDS - SINGLE DOCUMENT
# ---------------------------------------------------------

def get_top_words_for_document(
    tfidf_matrix,
    feature_names: np.ndarray,
    doc_index: int,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Find the most important words for one research paper.
    """

    empty_result = pd.DataFrame(
        columns=[
            "word",
            "tfidf_score"
        ]
    )

    # Safety checks
    if (
        tfidf_matrix is None
        or feature_names is None
        or len(feature_names) == 0
    ):
        return empty_result

    if tfidf_matrix.shape[0] == 0:
        return empty_result

    if (
        doc_index < 0
        or doc_index >= tfidf_matrix.shape[0]
    ):
        return empty_result

    # Get document TF-IDF row
    row = (
        tfidf_matrix[doc_index]
        .toarray()
        .flatten()
    )

    # Remove invalid values
    row = np.nan_to_num(
        row,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Highest TF-IDF scores
    top_indices = (
        row
        .argsort()[::-1][:top_n]
    )

    # Keep only words having a score
    top_indices = [
        index
        for index in top_indices
        if row[index] > 0
    ]

    result = pd.DataFrame({
        "word": feature_names[top_indices],
        "tfidf_score": np.round(
            row[top_indices],
            4
        )
    })

    return result


# ---------------------------------------------------------
# TF-IDF MATRIX EVALUATION
# ---------------------------------------------------------

def evaluate_tfidf_matrix(
    tfidf_matrix,
    feature_names: np.ndarray
) -> dict:
    """
    Calculate basic TF-IDF feature-space statistics.

    Returns:
        Number of documents
        Vocabulary size
        Sparsity percentage
        Average non-zero features per document
    """

    # Safety check
    if tfidf_matrix is None:
        return {
            "num_documents": 0,
            "vocabulary_size": 0,
            "sparsity_percent": 0,
            "avg_nonzero_features_per_doc": 0
        }

    n_docs, n_features = (
        tfidf_matrix.shape
    )

    n_nonzero = tfidf_matrix.nnz

    total_entries = (
        n_docs * n_features
    )

    if total_entries > 0:
        sparsity = (
            1 - (n_nonzero / total_entries)
        ) * 100
    else:
        sparsity = 0

    if n_docs > 0:
        avg_nonzero_per_doc = (
            n_nonzero / n_docs
        )
    else:
        avg_nonzero_per_doc = 0

    return {
        "num_documents": int(n_docs),

        "vocabulary_size": int(
            n_features
        ),

        "sparsity_percent": round(
            sparsity,
            2
        ),

        "avg_nonzero_features_per_doc": round(
            avg_nonzero_per_doc,
            2
        )
    }


# ---------------------------------------------------------
# MATPLOTLIB TF-IDF CHART
# ---------------------------------------------------------

def plot_top_words_overall(
    top_words_df: pd.DataFrame
):
    """
    Create a horizontal bar chart using Matplotlib.

    This intentionally avoids Streamlit's st.bar_chart(),
    which uses Vega/Altair and was causing the React #185
    rendering problem in the dashboard.
    """

    # Create empty chart if no data exists
    if (
        top_words_df is None
        or top_words_df.empty
    ):
        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.text(
            0.5,
            0.5,
            "No TF-IDF data available",
            ha="center",
            va="center",
            fontsize=12
        )

        ax.set_axis_off()

        fig.tight_layout()

        return fig

    # Make a copy
    plot_df = top_words_df.copy()

    # Remove invalid values
    plot_df = plot_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    plot_df = plot_df.dropna(
        subset=[
            "word",
            "avg_tfidf_score"
        ]
    )

    # Convert scores to numeric
    plot_df["avg_tfidf_score"] = pd.to_numeric(
        plot_df["avg_tfidf_score"],
        errors="coerce"
    )

    # Remove invalid rows again
    plot_df = plot_df.dropna(
        subset=["avg_tfidf_score"]
    )

    # If nothing remains
    if plot_df.empty:
        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.text(
            0.5,
            0.5,
            "No valid TF-IDF scores available",
            ha="center",
            va="center",
            fontsize=12
        )

        ax.set_axis_off()

        fig.tight_layout()

        return fig

    # Sort for horizontal bar chart
    plot_df = plot_df.sort_values(
        "avg_tfidf_score",
        ascending=True
    )

    # Create chart
    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    sns.barplot(
        data=plot_df,
        x="avg_tfidf_score",
        y="word",
        ax=ax
    )

    ax.set_title(
        "Top Important Words - Average TF-IDF",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Average TF-IDF Score"
    )

    ax.set_ylabel(
        "Research Terms"
    )

    fig.tight_layout()

    return fig