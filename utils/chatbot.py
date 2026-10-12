from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_chatbot_response(query, df):
    """
    A simple retrieval-based chatbot that finds the most relevant research paper
    matching the user's query using TF-IDF.
    """
    if not query or df.empty:
        return "I don't have any papers loaded or your query is empty.", 0.0, None
        
    # Combine title and abstract for matching
    docs = df['title'].fillna('') + " " + df['abstract'].fillna('')
    docs_list = docs.tolist()
    
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        X = vectorizer.fit_transform(docs_list + [query])
    except ValueError:
        return "I'm sorry, I couldn't understand that query.", 0.0, None
        
    query_vec = X[-1]
    doc_vecs = X[:-1]
    
    sim = cosine_similarity(query_vec, doc_vecs).flatten()
    best_idx = sim.argmax()
    best_score = sim[best_idx]
    
    if best_score > 0.05:
        paper = df.iloc[best_idx]
        title = paper['title']
        abstract = paper['abstract']
        
        response = f"I found a relevant paper for you!\n\n**{title}**\n\n*Summary snippet:* {str(abstract)[:300]}..."
        return response, best_score, paper
    else:
        return "I'm sorry, I couldn't find any papers matching your query. Could you try rephrasing or asking about a different topic?", 0.0, None
