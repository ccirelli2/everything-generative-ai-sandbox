"""

all-MiniLM-L6-v2
- It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for
tasks like clustering or semantic search.
- By default, input text longer than 256 word pieces is truncated.

================
References
================
1.https://pypi.org/project/vdblite/
2. Huggingface & Embeddings -> https://huggingface.co/blog/getting-started-with-embeddings
3. Library for creating vector embeddings -> https://www.sbert.net/index.html
4. all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
"""
import os
import sys
import vdblite
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from time import time
from uuid import uuid4
from pprint import pprint as pp
from more_itertools import batched


# Library Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# Globals
DIR_DATA = r"C:\Users\ccirelli\OneDrive - American International Group, Inc\Desktop\GitHub\generative-ai-sandbox\vector_databases\vd_light\data"
FILE_NAME = "moby10b.txt"
EMBEDDING_MODEL = EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Load Text File
mobyText = open(os.path.join(DIR_DATA, FILE_NAME), 'r').read()


def create_text_batches(text: str, num_tokens=256):
    """
    Create batches of text from a sentence.

    #>>> from more_itertools import batched
    #>>> text = "today is a good day to code.  Tomorrow may not be as good."
    #>>> batches = list(create_text_batches(text, num_tokens=5))
    #>>> assert len(batches)  == 3, f"batch len => {len(batches)}"
    """
    tokens = text.replace("\n", " ").split()
    batches = batched(tokens, num_tokens)
    return [' '.join(batch) for batch in batches]


if __name__ == '__main__':
    # Instantiate Instance of DB
    vdb = vdblite.Vdb()
    # Instantiate Embeddings Model
    embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME)
    # Create Batches of Text to Fit Embedding Size
    batches = create_text_batches(mobyText, num_tokens=256)[:1000]
    # Create Embeddings
    embeddings = embedding_model.encode(batches)
    # Generate metadata
    for b, v in zip(batches, embeddings):

        info = {'vector': v, 'sentence': b}
        # Append vector plus metadata to db
        vdb.add(info)

    # Create a Question
    question = "No fairy fingers can have pressed the gold, but devil's claws must have left their mouldings there since yesterday,"
    question_vector = embedding_model.encode(question)

    # Query the database
    response = vdb.search(
        vector=question_vector,
        field='vector',
        count=3
    )

    for r in response:
        print(r['score'], r['sentence'])
