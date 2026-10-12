import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def get_frequent_itemsets(texts, min_support=0.01, top_n=20):
    """
    Finds frequently co-occurring words (keywords) in a list of texts using a 
    co-occurrence matrix (a simple alternative to Apriori for Market Basket Analysis).
    """
    # Remove empty texts
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return pd.DataFrame()

    vectorizer = CountVectorizer(stop_words='english', binary=True, max_features=1000)
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return pd.DataFrame()
        
    # Co-occurrence matrix
    co_occurrence = (X.T * X)
    co_occurrence.setdiag(0)
    
    # Get vocabulary
    vocab = vectorizer.get_feature_names_out()
    
    # Extract edges
    cooc_matrix = co_occurrence.tocoo()
    
    edges = []
    num_docs = X.shape[0]
    for i, j, v in zip(cooc_matrix.row, cooc_matrix.col, cooc_matrix.data):
        if i < j: # only take upper triangle to avoid duplicates
            support = v / num_docs
            if support >= min_support:
                edges.append({
                    'Keyword 1': vocab[i],
                    'Keyword 2': vocab[j],
                    'Co-occurrences': v,
                    'Support': support
                })
                
    df = pd.DataFrame(edges)
    if not df.empty:
        df = df.sort_values(by='Co-occurrences', ascending=False).head(top_n)
        
    return df
