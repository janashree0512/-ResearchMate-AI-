from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

def answer_question(question, context):
    """
    A naive TF-IDF based Question Answering implementation.
    Finds the sentence in the context most similar to the question.
    """
    if not question or not context:
        return {"answer": "", "score": 0.0}
        
    try:
        # Tokenize context into sentences
        sentences = nltk.sent_tokenize(context)
        if not sentences:
            return {"answer": "No context available.", "score": 0.0}
            
        # Add question to the list of sentences to vectorize together
        documents = [question] + sentences
        
        # Calculate TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Calculate cosine similarity between question (idx 0) and all sentences (idx 1:)
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Find the index of the highest scoring sentence
        best_idx = cosine_similarities.argmax()
        best_score = cosine_similarities[best_idx]
        
        if best_score > 0:
            return {"answer": sentences[best_idx], "score": float(best_score)}
        else:
            return {"answer": "Could not find a relevant answer in the text.", "score": 0.0}
            
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "score": 0.0}
