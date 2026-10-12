from utils.ner import extract_named_entities_spacy, get_entity_frequencies
import pandas as pd

# 1. Test on a technical abstract
abstract = "Deep Reinforcement Learning (DRL) has revolutionized autonomous robot navigation. Mobile robots use these networks to traverse environments."
entities = extract_named_entities_spacy(abstract)
print("TEST 1 - Technical Abstract:")
print(get_entity_frequencies(entities))

# 2. Test on empty
print("\nTEST 2 - Empty Abstract:")
print(extract_named_entities_spacy(""))
