"""
ResearchMate AI - Intelligent Research Paper Analysis and Classification System
================================================================================
Run this app with:
    streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import nltk
import matplotlib.pyplot as plt

from utils.data_loader import load_dataset, get_full_text_series
from utils import eda
from utils import etl
from utils import preprocessing as prep
from utils import tfidf_extraction as tfidf_utils
from utils import summarization as summ_utils
from utils import classification as cls_utils
from utils import ner as ner_utils
from utils import qa as qa_utils
from utils import market_basket as mb_utils
from utils import chatbot as chat_utils


# ==========================================================================
# PAGE CONFIGURATION
# ==========================================================================

st.set_page_config(
    page_title="ResearchMate AI",
    page_icon="assets/icon.jpg",
    layout="wide",
)

DEFAULT_DATA_PATH = os.path.join("data", "sample_arxiv_data.csv")


# ==========================================================================
# NLTK SETUP
# ==========================================================================

@st.cache_resource
def ensure_nltk_data():
    """Make sure required NLTK resources are available (cloud-safe)."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("chunkers/maxent_ne_chunker", "maxent_ne_chunker"),
        ("corpora/words", "words"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)
    return True


ensure_nltk_data()


# ==========================================================================
# SIDEBAR
# ==========================================================================

# Sidebar logo + branding
_logo_path = os.path.join("assets", "logo.jpg")
_icon_path = os.path.join("assets", "icon.jpg")

if os.path.exists(_icon_path):
    st.sidebar.image(_icon_path, width=60)
st.sidebar.markdown("### ResearchMate AI")
st.sidebar.caption("Intelligent Research Paper Analysis")

st.sidebar.markdown("**Dataset Selection**")

uploaded_file = st.sidebar.file_uploader(
    "Upload your arXiv CSV (optional)",
    type=["csv"]
)

use_sample = st.sidebar.checkbox(
    "Use built-in sample dataset instead",
    value=uploaded_file is None
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Data Collection & EDA",
        "ETL Process",
        "Text Preprocessing & Feature Extraction",
        "Text Summarization",
        "Text Classification",
        "Named Entity Recognition",
        "Question Answering",
        "Market Basket Analysis",
        "Research Chatbot",
        "Evaluation & Results",
    ],
)


# ==========================================================================
# LOAD DATA
# ==========================================================================

@st.cache_data(show_spinner=False)
def _load(path_or_buffer):
    try:
        return load_dataset(path_or_buffer)
    except pd.errors.EmptyDataError:
        st.error("The provided dataset is empty or unreadable. Please check the file.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()


if uploaded_file is not None and not use_sample:
    df = _load(uploaded_file)
    data_source_label = f"Uploaded file: {uploaded_file.name}"
else:
    if not os.path.exists(DEFAULT_DATA_PATH):
        st.error(
            f"Sample dataset not found at {DEFAULT_DATA_PATH}. "
            "Please upload a CSV file instead."
        )
        st.stop()
    df = _load(DEFAULT_DATA_PATH)
    data_source_label = "Built-in sample arXiv dataset"


full_text_series = get_full_text_series(df)


# ==========================================================================
# PAGE: HOME
# ==========================================================================

if page == "Home":

    # Home page logo + title
    _logo_path = os.path.join("assets", "logo.jpg")
    if os.path.exists(_logo_path):
        col_logo, col_spacer = st.columns([2, 1])
        with col_logo:
            st.image(_logo_path, width=500)
    else:
        st.title("ResearchMate AI")
        st.subheader("Intelligent Research Paper Analysis and Classification System")

    st.markdown("---")

    # KPI CARDS
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Papers", len(df))
    col2.metric("Columns", len(df.columns))

    if "category" in df.columns:
        col3.metric("Categories", df["category"].nunique())
    else:
        col3.metric("Categories", "N/A")

    avg_words = full_text_series.fillna("").apply(lambda x: len(str(x).split())).mean()
    col4.metric("Avg Words", round(avg_words, 1))

    st.divider()

    st.subheader("Dataset Health")

    health_col1, health_col2, health_col3 = st.columns(3)

    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    health_col1.metric("Missing Values", missing_values)
    health_col2.metric("Duplicate Rows", duplicate_rows)
    health_col3.metric(
        "Data Source",
        "Uploaded CSV" if uploaded_file is not None and not use_sample else "Sample Dataset"
    )

    st.info(f"**Current data source:** {data_source_label}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Loaded {len(df)} research papers.")

    st.divider()

    st.subheader("Core Pipeline Steps")

    pipeline_col1, pipeline_col2, pipeline_col3 = st.columns(3)

    with pipeline_col1:
        st.info("Text Preprocessing")
        st.caption("Lowercase → punctuation removal → tokenization → stopword removal → lemmatization")

    with pipeline_col2:
        st.info("TF-IDF Vectorization")
        st.caption("Convert cleaned research text into numerical features and identify important terms.")

    with pipeline_col3:
        st.info("Summarization Models")
        st.caption("Rank sentences using TF-IDF and generate an extractive summary.")


# ==========================================================================
# PAGE: DATA COLLECTION & EDA
# ==========================================================================

elif page == "Data Collection & EDA":

    st.title("Data Collection & Data Loading")
    st.caption("Evaluation: 10 Marks Component")

    st.markdown(
        """
        The project uses the **arXiv Scientific Research Paper Dataset**.

        Each record contains:
        - Research paper title
        - Abstract
        - Category
        - Authors
        - Publication information

        You can upload your downloaded CSV file using the sidebar.
        """
    )

    st.success(f"Currently using: **{data_source_label}**")

    st.subheader("Raw Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Dataset Information")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Values", int(df.isna().sum().sum()))

    st.subheader("Column Data Types")
    st.dataframe(df.dtypes.astype(str).rename("Data Type"), use_container_width=True)

    # EDA Section inline
    st.divider()
    st.subheader("Exploratory Data Analysis")

    st.markdown(
        "EDA examines the structure and characteristics of the research paper dataset "
        "before applying NLP techniques."
    )

    shape_info = eda.get_shape_and_columns(df)
    c1, c2 = st.columns(2)
    c1.metric("Rows", shape_info["rows"])
    c2.metric("Columns", shape_info["columns"])
    st.write("**Columns:**", shape_info["column_names"])

    st.subheader("Missing Values & Duplicates")
    md_info = eda.get_missing_and_duplicates(df)
    st.dataframe(md_info["missing_table"], use_container_width=True)
    st.metric("Duplicate Rows", md_info["duplicate_rows"])

    if "category" in df.columns:
        st.subheader("Category Distribution")
        category_counts = eda.get_category_distribution(df)
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.dataframe(category_counts, use_container_width=True)
        with col2:
            fig = eda.plot_category_distribution(category_counts)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.subheader("Text Length Analysis")
    lengths = eda.compute_text_lengths(df)
    st.dataframe(lengths.describe().round(2), use_container_width=True)
    fig2 = eda.plot_text_length_distribution(lengths)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.subheader("Top Frequent Words (Raw Text)")
    st.caption("Computed on RAW, unprocessed text. Common stopwords are expected here.")
    top_n = st.slider("Number of top words to show", 5, 40, 20, key="eda_top_n")
    top_words_df = eda.get_top_frequent_words(full_text_series, top_n=top_n)
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.dataframe(top_words_df, use_container_width=True)
    with col2:
        fig3 = eda.plot_top_words(top_words_df)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    st.divider()
    st.subheader("Download Standardized Dataset")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Dataset (CSV)",
        csv_bytes,
        "standardized_arxiv_data.csv",
        "text/csv"
    )


# ==========================================================================
# PAGE: ETL
# ==========================================================================

elif page == "ETL Process":

    st.title("ETL Process")
    st.caption("Evaluation: 10 Marks Component")

    with st.expander("Understanding ETL vs EDA", expanded=True):
        st.markdown(
            """
            **EDA** examines and describes the dataset.

            **ETL** prepares the dataset for later processing.

            ### Extract
            Read the raw research paper dataset.

            ### Transform
            Clean the records, remove duplicates, handle missing values,
            combine text fields, and create useful derived columns.

            ### Load
            Save the transformed dataset to CSV and SQLite.
            """
        )

    st.header("Extract")
    st.caption("Reading the raw dataset from its source.")

    extracted_df = etl.extract_data(
        uploaded_file if (uploaded_file is not None and not use_sample) else DEFAULT_DATA_PATH
    )

    c1, c2 = st.columns(2)
    c1.metric("Source", data_source_label)
    c2.metric("Rows Extracted", len(extracted_df))
    st.dataframe(extracted_df.head(5), use_container_width=True)

    st.header("Transform")
    st.caption("Cleaning and reshaping the extracted records.")

    with st.spinner("Running transformation..."):
        transform_result = etl.transform_data(extracted_df.copy())

    transformed_df = transform_result["data"]
    report = transform_result["report"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows Before", report["rows_before"])
    c2.metric("Duplicates Removed", report["duplicates_removed"])
    c3.metric("Rows After", report["rows_after"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Missing Titles Filled", report["missing_title_filled"])
    c2.metric("Missing Abstracts Filled", report["missing_abstract_filled"])
    c3.metric("Missing Categories Filled", report["missing_category_filled"])

    st.write("**Columns added during transformation:**", report["columns_added"])

    st.subheader("Transformed Data Preview")
    st.dataframe(transformed_df.head(5), use_container_width=True)

    st.header("Load")
    st.caption("Persisting the transformed dataset to CSV and SQLite.")

    if st.button("Execute Load Step"):
        with st.spinner("Loading data into CSV and SQLite..."):
            csv_path = etl.load_to_csv(transformed_df)
            sqlite_info = etl.load_to_sqlite(transformed_df)
        st.success(f"Loaded {report['rows_after']} rows into CSV: `{csv_path}`")
        st.success(
            f"Loaded {sqlite_info['rows_loaded']} rows into SQLite table "
            f"`{sqlite_info['table_name']}` at `{sqlite_info['db_path']}`"
        )
    else:
        st.info("Click the button above to write the cleaned data to CSV and SQLite.")


# ==========================================================================
# PAGE: TEXT PREPROCESSING & FEATURE EXTRACTION
# ==========================================================================

elif page == "Text Preprocessing & Feature Extraction":

    st.title("Text Preprocessing & Feature Extraction")
    st.caption("Prerequisite Component")

    st.markdown(
        """
        The research paper text is cleaned using the following NLP pipeline:

        **Lowercasing → Punctuation Removal → Tokenization → Stopword Removal → Lemmatization**
        """
    )

    example_options = df["title"].fillna("(no title)").tolist()
    idx = st.selectbox(
        "Pick a research paper",
        options=range(len(df)),
        format_func=lambda i: example_options[i][:100],
        key="preprocess_doc_select"
    )

    original_text = str(full_text_series.iloc[idx])
    result = prep.preprocess_text(original_text)

    st.header("Before and After")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Original Text")
        st.text_area("Original", result["original"], height=250, label_visibility="collapsed")

    with c2:
        st.subheader("Cleaned Text")
        st.text_area("Cleaned", result["cleaned_text"], height=250, label_visibility="collapsed")

    st.header("Preprocessing Evaluation")
    original_tokens = result["tokens"]
    cleaned_tokens = result["lemmatized_tokens"]
    original_count = len(original_tokens)
    cleaned_count = len(cleaned_tokens)
    removed_count = max(original_count - cleaned_count, 0)
    reduction = (removed_count / original_count * 100 if original_count > 0 else 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original Tokens", original_count)
    c2.metric("Clean Tokens", cleaned_count)
    c3.metric("Tokens Removed", removed_count)
    c4.metric("Reduction", f"{reduction:.2f}%")

    st.header("Preprocessing Steps")
    step1, step2, step3, step4 = st.columns(4)
    with step1:
        st.subheader("1. Lowercase")
        st.write(result["lowercased"][:1000])
    with step2:
        st.subheader("2. Punctuation Removal")
        st.write(result["no_punctuation"][:1000])
    with step3:
        st.subheader("3. Tokenization")
        st.write(result["tokens"][:50])
    with step4:
        st.subheader("4. Lemmatization")
        st.write(result["lemmatized_tokens"][:50])

    st.success("Text preprocessing completed successfully.")

    st.header("Corpus-level Before vs After")
    with st.spinner("Calculating preprocessing statistics..."):
        cleaned_series = prep.preprocess_series(full_text_series)
        st.session_state["cleaned_text_series"] = cleaned_series
        stats_df = prep.compute_before_after_stats(full_text_series, cleaned_series)
        st.dataframe(stats_df, use_container_width=True)

    # TF-IDF Section
    st.divider()
    st.header("TF-IDF Feature Extraction")

    st.markdown(
        """
        **TF-IDF (Term Frequency - Inverse Document Frequency)** converts
        research paper text into numerical features.

        - **TF** measures how often a word appears in a document.
        - **IDF** measures how uncommon the word is across documents.
        - A high TF-IDF score indicates an important distinguishing term.
        """
    )

    if "cleaned_text_series" in st.session_state:
        cleaned_series = st.session_state["cleaned_text_series"]
    else:
        cleaned_series = prep.preprocess_series(full_text_series)
        st.session_state["cleaned_text_series"] = cleaned_series

    max_features = st.slider("Maximum vocabulary size", 100, 5000, 2000, step=100)

    try:
        vectorizer, tfidf_matrix, feature_names = tfidf_utils.fit_tfidf(
            cleaned_series, max_features=max_features
        )
    except ValueError as e:
        st.error(f"TF-IDF could not be calculated: {e}")
        st.stop()

    st.subheader("TF-IDF Feature Space")
    eval_metrics = tfidf_utils.evaluate_tfidf_matrix(tfidf_matrix, feature_names)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", eval_metrics["num_documents"])
    c2.metric("Vocabulary Size", eval_metrics["vocabulary_size"])
    c3.metric("Sparsity", f"{eval_metrics['sparsity_percent']}%")
    c4.metric("Avg Features / Doc", eval_metrics["avg_nonzero_features_per_doc"])
    st.caption("High sparsity is normal for TF-IDF because most words do not appear in every document.")

    st.subheader("Top Important Words (Corpus-wide)")
    top_n = st.slider("Number of top words", 5, 50, 20, key="tfidf_top_n")
    top_words_df = tfidf_utils.get_top_words_overall(tfidf_matrix, feature_names, top_n=top_n)
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.dataframe(top_words_df, use_container_width=True)
    with col2:
        if not top_words_df.empty:
            fig_tfidf = tfidf_utils.plot_top_words_overall(top_words_df)
            st.pyplot(fig_tfidf, use_container_width=True)
            plt.close(fig_tfidf)
        else:
            st.warning("No TF-IDF terms are available for visualization.")

    st.subheader("Top Important Words for a Single Document")
    example_options_tfidf = df["title"].fillna("(no title)").tolist()
    idx2 = st.selectbox(
        "Pick a paper",
        options=range(len(df)),
        format_func=lambda i: example_options_tfidf[i][:80],
        key="tfidf_doc_select"
    )
    try:
        doc_top_words = tfidf_utils.get_top_words_for_document(
            tfidf_matrix, feature_names, idx2, top_n=10
        )
        st.write(f"**Title:** {df.iloc[idx2]['title']}")
        st.dataframe(doc_top_words, use_container_width=True)
    except Exception as e:
        st.error(f"Unable to calculate document-level TF-IDF: {e}")


# ==========================================================================
# PAGE: TEXT SUMMARIZATION
# ==========================================================================

elif page == "Text Summarization":

    st.title("Extractive Text Summarization")
    st.caption("Phase I — Concept 1")

    st.markdown(
        """
        The summarizer uses **TF-IDF sentence ranking**.

        1. Split the abstract into sentences.
        2. Calculate word importance using TF-IDF.
        3. Score each sentence by summing its word scores.
        4. Select the highest-scoring sentences.
        5. Reassemble them in their original order.
        """
    )

    example_options = df["title"].fillna("(no title)").tolist()
    idx = st.selectbox(
        "Pick a paper to summarize",
        options=range(len(df)),
        format_func=lambda i: example_options[i][:80],
        key="summary_doc_select"
    )

    num_sentences = st.slider("Number of sentences in summary", 1, 5, 2, key="summary_sentence_count")

    original_text = (
        str(df.iloc[idx]["abstract"]) if pd.notna(df.iloc[idx]["abstract"]) else ""
    )

    st.subheader("Original Abstract")
    if original_text:
        st.write(original_text)
    else:
        st.warning("No abstract available for this paper.")

    if original_text:
        result = summ_utils.summarize_text(original_text, num_sentences=num_sentences)

        st.subheader("Generated Summary")
        st.success(result["summary"])

        st.subheader("Compression Evaluation")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Sentences", result["sentences_total"])
        c2.metric("Sentences Selected", result["sentences_selected"])
        c3.metric("Compression Ratio", result["compression_ratio"])
        c4.metric("Reduction", f"{result['reduction_percent']}%")
        st.caption(
            f"Original word count: {result['original_word_count']} → "
            f"Summary word count: {result['summary_word_count']}"
        )

        st.subheader("ROUGE Evaluation")
        rouge_scores = summ_utils.evaluate_summary_with_rouge(original_text, result["summary"])
        c1, c2, c3 = st.columns(3)
        c1.metric("ROUGE-1 F1", rouge_scores["rouge1_f1"])
        c2.metric("ROUGE-2 F1", rouge_scores["rouge2_f1"])
        c3.metric("ROUGE-L F1", rouge_scores["rougeL_f1"])
        st.caption(
            "ROUGE-1 measures single-word overlap, ROUGE-2 measures two-word phrase overlap, "
            "and ROUGE-L measures longest common subsequence overlap."
        )

        st.divider()
        st.subheader("Batch Summarization Preview")
        st.caption("Summarizing the first 5 papers for a quick comparison.")

        batch_rows = []
        for i in range(min(5, len(df))):
            text = str(df.iloc[i]["abstract"]) if pd.notna(df.iloc[i]["abstract"]) else ""
            if not text:
                continue
            res = summ_utils.summarize_text(text, num_sentences=num_sentences)
            rouge = summ_utils.evaluate_summary_with_rouge(text, res["summary"])
            batch_rows.append({
                "title": df.iloc[i]["title"],
                "original_words": res["original_word_count"],
                "summary_words": res["summary_word_count"],
                "reduction_%": res["reduction_percent"],
                "rouge1_f1": rouge["rouge1_f1"],
                "rougeL_f1": rouge["rougeL_f1"],
            })

        if batch_rows:
            st.dataframe(pd.DataFrame(batch_rows), use_container_width=True)
        else:
            st.info("No papers with abstracts are available for batch summarization.")


# ==========================================================================
# PAGE: TEXT CLASSIFICATION
# ==========================================================================

elif page == "Text Classification":

    st.title("Text Classification")
    st.caption("Phase I — Concept 2")

    st.markdown(
        """
        **Text Classification** is used to predict the category of a research paper
        based on its abstract. We use a machine learning model trained on TF-IDF features.
        """
    )

    if "category" not in df.columns:
        st.error("The dataset does not have a 'category' column for classification.")
    else:
        model_type = st.radio(
            "Select Model",
            ["Multinomial Naive Bayes (nb)", "Logistic Regression (lr)"],
            format_func=lambda x: x.split(" (")[0]
        )
        model_key = "lr" if "Logistic Regression" in model_type else "nb"

        if st.button("Train & Evaluate Model"):
            with st.spinner("Training model..."):
                try:
                    res = cls_utils.train_and_evaluate_classifier(
                        df, text_col="abstract", target_col="category", model_type=model_key
                    )
                    st.success(f"Model trained successfully! Accuracy: {res['accuracy']:.4f}")

                    st.subheader("Classification Report")
                    report_df = pd.DataFrame(res["report"]).transpose()
                    st.dataframe(report_df.style.format(precision=3), use_container_width=True)

                    st.subheader("Confusion Matrix")
                    fig_cm = cls_utils.plot_confusion_matrix(res["confusion_matrix"], res["classes"])
                    st.pyplot(fig_cm, use_container_width=True)
                except Exception as e:
                    st.error(f"Error during classification: {e}")


# ==========================================================================
# PAGE: NAMED ENTITY RECOGNITION
# ==========================================================================

elif page == "Named Entity Recognition":

    st.title("Named Entity Recognition (NER)")
    st.caption("Phase II — Concept 3")

    st.markdown(
        """
        **Named Entity Recognition (NER)** identifies and classifies key entities
        in the text such as Organizations, Persons, Locations, and technical concepts.
        """
    )

    example_options = df["title"].fillna("(no title)").tolist()
    idx = st.selectbox(
        "Pick a paper to analyze",
        options=range(len(df)),
        format_func=lambda i: example_options[i][:80],
        key="ner_doc_select"
    )

    text = str(df.iloc[idx]["abstract"]) if pd.notna(df.iloc[idx]["abstract"]) else ""

    st.subheader("Original Abstract")
    st.write(text)

    if st.button("Extract Entities"):
        with st.spinner("Extracting..."):
            try:
                entities = ner_utils.extract_named_entities_spacy(text)
                if not entities:
                    st.warning("No named entities or technical concepts could be extracted from this text.")
                    st.info(
                        "**Debugging Information:**\n"
                        "- The abstract might be too short or completely devoid of recognized organizations, people, or locations.\n"
                        "- Technical terms must be at least two words long (e.g., 'Deep Reinforcement Learning') to be classified as a CONCEPT.\n"
                        "- Check if the abstract column is missing or empty."
                    )
                else:
                    freq_df = ner_utils.get_entity_frequencies(entities)
                    st.subheader("Extracted Entities")
                    st.dataframe(freq_df, use_container_width=True)
            except ImportError as ie:
                st.error(f"{str(ie)}")
                st.info("Run `pip install spacy` in your terminal.")
            except OSError as oe:
                st.error(f"{str(oe)}")
                st.info("Run `python -m spacy download en_core_web_sm` in your terminal.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")


# ==========================================================================
# PAGE: QUESTION ANSWERING
# ==========================================================================

elif page == "Question Answering":

    st.title("Question Answering (QA)")
    st.caption("Phase II — Concept 4")

    st.markdown(
        """
        **Question Answering** allows you to ask a question in natural language
        and extract the most relevant sentence from the research paper abstract.

        *(Note: This uses an extractive TF-IDF similarity approach to operate
        fast and without heavy deep learning models.)*
        """
    )

    example_options = df["title"].fillna("(no title)").tolist()
    idx = st.selectbox(
        "Pick a paper as context",
        options=range(len(df)),
        format_func=lambda i: example_options[i][:80],
        key="qa_doc_select"
    )

    context = str(df.iloc[idx]["abstract"]) if pd.notna(df.iloc[idx]["abstract"]) else ""

    st.subheader("Context (Abstract)")
    st.write(context)

    question = st.text_input("Ask a question about this paper:")

    if st.button("Get Answer") and question:
        with st.spinner("Finding answer..."):
            ans = qa_utils.answer_question(question, context)
            if "Error" in ans.get("answer", ""):
                st.error(ans["answer"])
            else:
                st.success(f"**Answer:** {ans['answer']}")
                st.caption(f"Confidence score: {ans['score']:.4f}")


# ==========================================================================
# PAGE: MARKET BASKET ANALYSIS
# ==========================================================================

elif page == "Market Basket Analysis":

    st.title("Market Basket Analysis")
    st.caption("Final Phase — Concept 5")

    st.markdown(
        """
        **Market Basket Analysis** on text evaluates which keywords frequently appear
        together in the same research paper. This uses keyword co-occurrence matrices
        computed across all paper abstracts.
        """
    )

    min_support = st.slider("Minimum Support", 0.01, 0.50, 0.05, 0.01)

    if st.button("Run Keyword Co-occurrence Analysis"):
        with st.spinner("Analyzing keyword associations..."):
            texts = df["abstract"].fillna("").tolist()
            res_df = mb_utils.get_frequent_itemsets(texts, min_support=min_support, top_n=30)

        if res_df.empty:
            st.warning("No associations found. Try lowering the minimum support.")
        else:
            st.success("Analysis complete!")
            st.dataframe(res_df, use_container_width=True)


# ==========================================================================
# PAGE: RESEARCH CHATBOT
# ==========================================================================

elif page == "Research Chatbot":

    st.title("Research Assistant Chatbot")
    st.caption("Final Phase — Concept 6")

    st.markdown(
        """
        Ask questions about research topics covered in the dataset.
        The chatbot uses TF-IDF similarity to retrieve the most relevant paper.
        """
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask me about research topics in the dataset...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, score, paper = chat_utils.get_chatbot_response(query, df)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


# ==========================================================================
# PAGE: EVALUATION & RESULTS
# ==========================================================================

elif page == "Evaluation & Results":

    st.title("Evaluation & Results")

    st.markdown(
        """
        This dashboard maps the project implementation to the evaluation criteria
        for Internal Evaluation II and CAT II.
        """
    )

    st.subheader("Evaluation Rubric Mapping")

    rubric_data = [
        {
            "Criteria": "Data Collection & Pre-processing",
            "Marks": "10 marks",
            "Status": "Completed",
            "Evidence": "See 'Data Collection & EDA' and 'Text Preprocessing & Feature Extraction'.",
        },
        {
            "Criteria": "ETL Process",
            "Marks": "10 marks",
            "Status": "Completed",
            "Evidence": "See 'ETL Process' page for extraction, cleaning, and loading.",
        },
        {
            "Criteria": "Phase I — First 2 Concepts",
            "Marks": "20 marks",
            "Status": "Completed",
            "Evidence": "Concept 1 (Text Summarization) & Concept 2 (Text Classification) are fully functional.",
        },
        {
            "Criteria": "Phase II — 3rd and 4th Concepts",
            "Marks": "20 marks",
            "Status": "Completed",
            "Evidence": "Concept 3 (Named Entity Recognition) & Concept 4 (Question Answering) are fully functional.",
        },
        {
            "Criteria": "Final Phase — 5th and 6th Concepts",
            "Marks": "Required",
            "Status": "Completed",
            "Evidence": "Concept 5 (Market Basket Analysis) & Concept 6 (Research Chatbot) are fully functional.",
        },
    ]

    st.dataframe(pd.DataFrame(rubric_data), use_container_width=True)

    st.divider()
    st.subheader("Concepts Implemented")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Phase I**")
        st.markdown("- Text Summarization (Extractive, TF-IDF sentence ranking)")
        st.markdown("- Text Classification (Naive Bayes / Logistic Regression on TF-IDF)")
        st.markdown("**Phase II**")
        st.markdown("- Named Entity Recognition (spaCy `en_core_web_sm`)")
        st.markdown("- Question Answering (TF-IDF cosine similarity extraction)")
    with c2:
        st.markdown("**Final Phase**")
        st.markdown("- Market Basket Analysis (keyword co-occurrence matrix)")
        st.markdown("- Research Chatbot (TF-IDF retrieval-based chatbot)")
        st.markdown("**Supporting Components**")
        st.markdown("- Data Collection & EDA (arXiv CSV loader + statistical analysis)")
        st.markdown("- ETL Process (extract, clean, load to CSV & SQLite)")
        st.markdown("- Text Preprocessing & TF-IDF Feature Extraction")

    st.success("All evaluation requirements have been met. The dashboard is ready for demonstration.")