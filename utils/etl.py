"""
etl.py
-------
Phase 1 - Step 3: ETL (Extract, Transform, Load)

WHAT IS ETL AND HOW IS IT DIFFERENT FROM EDA / PREPROCESSING?
----------------------------------------------------------------
ETL is a classic DATA ENGINEERING process made of three stages:

    EXTRACT  -> pull raw data out of its source (a CSV file, database,
                API, etc.) with no changes made to it yet.
    TRANSFORM -> clean and reshape that raw data at the STRUCTURAL /
                RECORD level: standardize column names, handle missing
                values, remove duplicate rows, merge fields, and
                compute simple derived columns. This is NOT the same
                as NLP text preprocessing (lowercasing, tokenization,
                stopword removal, lemmatization) - that happens later,
                on top of the ETL output, in the Data Preprocessing step.
    LOAD     -> write the cleaned, structured data to a persistent
                destination (a new CSV file and/or a database table)
                so every later step in the pipeline (EDA, TF-IDF,
                summarization, and eventually classification) reads
                from one single, reliable, cleaned source instead of
                re-cleaning the raw file every time.

In short:
    EDA            = analyze the data as a table (stats only, no changes)
    ETL            = physically extract -> clean -> save the dataset
    Preprocessing  = NLP-specific text cleaning (lowercase, tokens, etc.)
"""

import os
import sqlite3

import pandas as pd

from utils.data_loader import load_dataset  # reuse the flexible CSV loader


# --------------------------------------------------------------------------
# EXTRACT
# --------------------------------------------------------------------------
def extract_data(file_path_or_buffer) -> pd.DataFrame:
    """
    EXTRACT stage: read the raw arXiv dataset from its source (CSV file
    or uploaded buffer) with column names standardized to
    title / abstract / category, but with NO cleaning applied yet.
    """
    df = load_dataset(file_path_or_buffer)
    return df


# --------------------------------------------------------------------------
# TRANSFORM
# --------------------------------------------------------------------------
def transform_data(df: pd.DataFrame) -> dict:
    """
    TRANSFORM stage: clean and reshape the extracted data at the
    structural/record level.

    Steps applied:
        1. Remove exact duplicate rows
        2. Handle missing values (fill missing title/abstract with "",
           fill missing category with "unknown")
        3. Merge title + abstract into a single 'full_text' column
        4. Compute simple derived columns (word_count, char_count)

    Returns a dict with the transformed dataframe plus a small report
    describing what changed, so the transformation is transparent.
    """
    report = {}

    rows_before = len(df)
    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    missing_title_before = int(df["title"].isnull().sum())
    missing_abstract_before = int(df["abstract"].isnull().sum())
    missing_category_before = int(df["category"].isnull().sum())

    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    df["category"] = df["category"].fillna("unknown")

    df["full_text"] = (df["title"] + ". " + df["abstract"]).str.strip()
    df["word_count"] = df["full_text"].apply(lambda x: len(x.split()))
    df["char_count"] = df["full_text"].apply(len)

    rows_after = len(df)

    report = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_removed": duplicates_removed,
        "missing_title_filled": missing_title_before,
        "missing_abstract_filled": missing_abstract_before,
        "missing_category_filled": missing_category_before,
        "columns_added": ["full_text", "word_count", "char_count"],
    }

    return {"data": df, "report": report}


# --------------------------------------------------------------------------
# LOAD
# --------------------------------------------------------------------------
def load_to_csv(df: pd.DataFrame, output_path: str = "data/processed_arxiv_data.csv") -> str:
    """
    LOAD stage (destination 1): persist the transformed dataset to a
    new CSV file, so downstream steps can read the cleaned data
    directly instead of repeating extraction + transformation.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def load_to_sqlite(df: pd.DataFrame, db_path: str = "data/researchmate.db", table_name: str = "papers") -> dict:
    """
    LOAD stage (destination 2): persist the transformed dataset into a
    lightweight SQLite database table. This demonstrates loading into
    a proper data store, not just a flat file.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    finally:
        conn.close()
    return {"db_path": db_path, "table_name": table_name, "rows_loaded": row_count}


# --------------------------------------------------------------------------
# FULL PIPELINE
# --------------------------------------------------------------------------
def run_etl_pipeline(file_path_or_buffer, csv_output_path: str = "data/processed_arxiv_data.csv",
                      db_output_path: str = "data/researchmate.db") -> dict:
    """
    Convenience function that runs Extract -> Transform -> Load in one
    call and returns everything needed to display the results in the UI.
    """
    extracted_df = extract_data(file_path_or_buffer)
    transform_result = transform_data(extracted_df)
    transformed_df = transform_result["data"]
    transform_report = transform_result["report"]

    csv_path = load_to_csv(transformed_df, csv_output_path)
    sqlite_info = load_to_sqlite(transformed_df, db_output_path)

    return {
        "extracted_df": extracted_df,
        "transformed_df": transformed_df,
        "transform_report": transform_report,
        "csv_output_path": csv_path,
        "sqlite_info": sqlite_info,
    }