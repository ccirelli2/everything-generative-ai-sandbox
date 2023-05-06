"""
Notes:
    - Embedding Model: 'all-MiniLM-L6-v2'
        - max_seq_length = 256.  This means that the maximum number of tokens
        that can be encoded into a single vector embedding is 256.
        Anything beyond this must be truncated.
        - word_embedding_dimension = 384.  This is the number of dims
        of the vector output by the model.
        - Normalize(): noramlized output vector so that we can use dot product
        similarity metric.


References
===================
- https://docs.pinecone.io/docs/quickstart
- https://colab.research.google.com/github/pinecone-io/examples/blob/master/quick-tour/hello-pinecone.ipynb#scrollTo=indirect-lafayette
"""

# Import Libraries
import os
import pandas as pd
import pinecone
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# Create Connectionw/ Pinecone
print('Instantiating connection w/ Pinecone')
pinecone.init(
        api_key="86e1d60e-8c7c-45f5-966b-7309d62e7ecf",
        environment="us-west1-gcp-free"
)


# Assign a name to the index
print('Creating Index Name')
index_name = "hello-pinecone"

if index_name in pinecone.list_indexes():
    print('deleting index')
    pinecone.delete_index(index_name)
    
else:
    # Create Index
    print('Creating Index')
    dimensions = 8
    pinecone.create_index(name=index_name, dimension=dimensions, metric="cosine")
    print('Finished creating index')


# Get list of Index
print(f"List of available indexes => {pinecone.list_indexes()}")


# Connect to the index
print(f"Connecting to index => {index_name}")
index = pinecone.Index(index_name)

# Load Dataset
"""
questions format
- list of dictionary objects.
- dictionary keys are
    'id' whose value is a list [#,#]
    'text': list of questions.

"""
print("Downloading test dataset")
dataset = load_dataset('quora', split='train')
questions = dataset['questions'][:2]


# Build Embeddings
print("Instantiating Embedding Model")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Encode Sentences
"""
Upsert vector must be of form [(id, embedded vector, metadata)]
and metadata is {'key': value}
"""
"""
print("Encoding Questions")
vectors = []
for i in tqdm(range(len(questions))):
    q = questions[i]['text'][0]
    xq = model.encode(q)
    meta = {i: q}
    v = (i, xq, meta)
    vectors.append(vectors)

print(vectors)

# Upsert Vectors
index.upsert(vectors=vectors)
"""

print(f"Upserting Vectors")
index.upsert([
    ("A", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    ("B", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
    ("C", [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
    ("D", [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
    ("E", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
])
print(f"Upsert completed")




