import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate_classifier(df, text_col, target_col, model_type='nb', test_size=0.2, max_features=5000):
    """
    Trains a classification model and evaluates it.
    """
    # Drop missing values
    df_clean = df.dropna(subset=[text_col, target_col]).copy()
    
    # Get features and targets
    X = df_clean[text_col]
    y = df_clean[target_col]
    
    # Check if we have enough data and classes
    if len(X) < 10 or len(y.unique()) < 2:
        raise ValueError("Not enough data or classes for classification.")
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Vectorization
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Model
    if model_type == 'lr':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        model = MultinomialNB()
        
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    classes = np.unique(y)
    
    return {
        'model': model,
        'vectorizer': vectorizer,
        'accuracy': acc,
        'report': report,
        'confusion_matrix': cm,
        'classes': classes,
        'y_test': y_test,
        'y_pred': y_pred
    }

def plot_confusion_matrix(cm, classes):
    """
    Plots a confusion matrix using seaborn.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return fig
