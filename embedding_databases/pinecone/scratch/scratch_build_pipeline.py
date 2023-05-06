"""
References
=========================
- https://huggingface.co/sentence-transformers
"""

# Import libraries
import os
import pinecone
from decouple import config
import logging

# Directories
ROOT_DIR = "/home/oem/repositories/generative-ai-sandbox"
os.chdir(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'gensim_sandbox/data')

# Project Packages
import embedding_databases.pinecone.src.pipeline as cd

# Globals
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_FILENAME = "moby_dick.txt"
SENTENCES = open(os.path.join(DATA_DIR, TEXT_FILENAME), 'r').readlines()


if __name__ == "__main__":
    vectorize = cd.CreateEmbeddings(
        sentences=SENTENCES,
        transformer_name=MODEL_NAME
    )

    # Vectorize Sentences
    vectorize.embed()
    print(len(vectorize.embeddings))





