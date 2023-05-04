"""
Script to learn how to upsert data into pinecone.
"""
import os
import pinecone
import pandas as pd
from decouple import config


# Globals
API_KEY = config('API_KEY')
ENV_NAME = config('ENVIRONMENT')

# Instantiate connection to pinecone
print("Instantiating connection to pinecone\n")
pinecone.init(api_key=API_KEY, environment=ENV_NAME)

# Create Index
index_name = 'hello-pinecone'
index = pinecone.Index(index_name)

# Check on Insertions
index_stats = index.describe_index_stats()
print(index_stats)

# Query Index
print(f"Querying index => {index_name}\n")
vector=[0.3 for x in range(8)]
"""
query_response = index.query(
        vector=vector,
        top_k=1,
        include_values=True
    )

print(f"Query response => {query_response}\n")
"""

# Using Metadata filters in queries
"""
You can add metadata to document embeddings within Pinecone and then
filter for those criteria when sending the query.  Pinecone will search
for similar vector embeddings only among those items that match the filter.

Ref: https://docs.pinecone.io/docs/query-data
"""
query_w_meta = index.query(
        vector=vector,
        filter={"genre": "comedy", "year": 2020},
        top_k=1,
        include_metadata=True
        )

print(f"Query response => {query_w_meta}\n")

