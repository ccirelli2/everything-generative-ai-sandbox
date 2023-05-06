"""
References
=============================
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
"""

# Import Libraries
from sentence_transformers import SentenceTransformer


# Tests Sentences
sentences = ["This is an example sentence", "Each sentence is converted"]

# Instantiate Model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Convert Sentences to Embeddings
embeddings = model.encode(sentences)

# Print Embeddings
for e in embeddings:
    print(len(e))



