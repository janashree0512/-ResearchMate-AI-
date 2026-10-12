import pandas as pd
import collections

def extract_named_entities_spacy(text):
    """
    Extracts named entities from text using spaCy.
    Includes a fallback/enhancement to extract technical terminology 
    (Noun Phrases) as CONCEPT entities for research papers.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        import spacy
    except ImportError:
        raise ImportError("spaCy is not installed. Please install it using `pip install spacy`.")

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise OSError("spaCy model 'en_core_web_sm' not found. Please install it using `python -m spacy download en_core_web_sm`.")
        
    doc = nlp(text)
    
    entities = []
    # Track the span indices of entities so we don't overlap with noun chunks
    entity_spans = set()
    
    # 1. Standard Entities (PERSON, ORG, GPE, etc.)
    for ent in doc.ents:
        if ent.text.strip():
            entities.append({'text': ent.text.strip(), 'label': ent.label_})
            for i in range(ent.start, ent.end):
                entity_spans.add(i)
                
    # 2. Technical terminology (Noun chunks as CONCEPT)
    for chunk in doc.noun_chunks:
        # Check if it overlaps with an existing entity
        overlap = any(i in entity_spans for i in range(chunk.start, chunk.end))
        if not overlap:
            # A valid technical term is usually > 1 word (excluding stopwords)
            clean_text = " ".join([token.text for token in chunk if not token.is_stop and not token.is_punct])
            if clean_text and len(clean_text.split()) > 1:
                # Capitalize words like "Deep Reinforcement Learning"
                clean_text = clean_text.title() 
                entities.append({'text': clean_text, 'label': 'CONCEPT'})
                for i in range(chunk.start, chunk.end):
                    entity_spans.add(i)
            
    return entities

def get_entity_frequencies(entities):
    """
    Returns a DataFrame of entity frequencies.
    """
    if not entities:
        return pd.DataFrame()
        
    entity_counts = collections.Counter([(e['text'], e['label']) for e in entities])
    df = pd.DataFrame(entity_counts.items(), columns=['EntityTuple', 'Count'])
    df[['Entity', 'Entity Type']] = pd.DataFrame(df['EntityTuple'].tolist(), index=df.index)
    df = df[['Entity', 'Entity Type', 'Count']].sort_values(by='Count', ascending=False)
    # Return formatted table
    return df[['Entity', 'Entity Type', 'Count']]
