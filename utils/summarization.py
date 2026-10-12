"""
summarization.py
------------------
NLP Concept 3: Extractive Text Summarization using TF-IDF Sentence Ranking

How it works:
    1. Split the document into sentences.
    2. Compute a TF-IDF score for every word in the document.
    3. Score each sentence by summing the TF-IDF scores of the words
       it contains (normalized by sentence length so long sentences
       don't win purely on word count).
    4. Select the top-N highest scoring sentences and reassemble them
       IN THEIR ORIGINAL ORDER to form the summary ("extractive"
       because sentences are extracted verbatim, not rewritten).

Evaluation:
    - Compression ratio & reduction %: how much shorter the summary is.
    - ROUGE-1 / ROUGE-2 / ROUGE-L: measures the overlap between the
      generated summary and the ORIGINAL text (used here as the
      reference, since this beginner dataset has no separately
      human-written "gold" summary). This tells us how much of the
      original content's key wording was preserved by the extracted
      sentences.
"""

import re

import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer

STOPWORDS = set(stopwords.words("english"))


def summarize_text(text: str, num_sentences: int = 2) -> dict:
    """
    Generate an extractive summary of `text` using TF-IDF sentence
    ranking, and compute compression statistics.
    """
    if not isinstance(text, str) or text.strip() == "":
        return {
            "original_text": text,
            "summary": "",
            "sentences_total": 0,
            "sentences_selected": 0,
            "original_word_count": 0,
            "summary_word_count": 0,
            "compression_ratio": 0.0,
            "reduction_percent": 0.0,
        }

    sentences = sent_tokenize(text)

    # If the document is already very short, there's nothing meaningful to compress
    if len(sentences) <= num_sentences:
        summary = text
        selected_count = len(sentences)
    else:
        # Fit TF-IDF at the SENTENCE level (each sentence = one "document")
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            # happens if all sentences are empty/stopwords only
            summary = " ".join(sentences[:num_sentences])
            selected_count = num_sentences
            return _build_summary_result(text, summary, len(sentences), selected_count)

        # Sentence score = average TF-IDF weight of its words
        sentence_scores = np.asarray(tfidf_matrix.mean(axis=1)).flatten()

        # Pick top-N sentence indices, then restore original order
        top_indices = sentence_scores.argsort()[::-1][:num_sentences]
        top_indices_sorted = sorted(top_indices)
        summary_sentences = [sentences[i] for i in top_indices_sorted]
        summary = " ".join(summary_sentences)
        selected_count = len(summary_sentences)

    return _build_summary_result(text, summary, len(sentences), selected_count)


def _build_summary_result(original_text: str, summary: str, total_sentences: int, selected_sentences: int) -> dict:
    original_words = len(word_tokenize(original_text))
    summary_words = len(word_tokenize(summary)) if summary else 0

    compression_ratio = round(summary_words / original_words, 3) if original_words else 0.0
    reduction_percent = round((1 - compression_ratio) * 100, 2) if original_words else 0.0

    return {
        "original_text": original_text,
        "summary": summary,
        "sentences_total": total_sentences,
        "sentences_selected": selected_sentences,
        "original_word_count": original_words,
        "summary_word_count": summary_words,
        "compression_ratio": compression_ratio,
        "reduction_percent": reduction_percent,
    }


def evaluate_summary_with_rouge(original_text: str, summary: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores comparing the
    generated summary against the original text (used as the
    reference "gold" text in this beginner setup).

    ROUGE-1: unigram (single word) overlap
    ROUGE-2: bigram (two consecutive words) overlap
    ROUGE-L: longest common subsequence overlap (captures sentence
             structure similarity, not just word overlap)
    """
    if not summary.strip():
        return {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(original_text, summary)

    return {
        "rouge1_f1": round(scores["rouge1"].fmeasure, 4),
        "rouge2_f1": round(scores["rouge2"].fmeasure, 4),
        "rougeL_f1": round(scores["rougeL"].fmeasure, 4),
    }
