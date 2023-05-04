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

# List Indexes
indexes = pinecone.list_indexes()
collections = pinecone.list_collections()
print(f"Indexes => {indexes}")
print(f"Collections => {collections}\n")

# Create Index
index_name = 'hello-pinecone'
index_dim = 8
index_metric = 'cosine'
recreat_index = False

if index_name in indexes:
    print(f"Index {index_name} exists")
    
    if recreat_index:
        print(f"Deleting index => {index_name}\n")
        pinecone.delete_index(index_name)
        print(f"Creating index => {index_name}")
        index = pinecone.create_index(
                name=index_name,
                dimension=index_dim,
                metric=index_metric
            )
        print(f"Process complete")
    
    else:
        index = pinecone.Index(index_name)

else:
    print(f"Creating index => {index_name}")
    index = pinecone.create_index(
            name=index_name,
            dimension=index_dim,
            metric=index_metric
        )
    print(f"Process complete")



# Create Data with Metadata
'''insert data as a list of tuples [(id, vector)]
    metadata: add dictionary after vector.
'''
data = [
    ("vec1",
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        {"genre": "comedy", "year": 2020}
    ),
    ("vec2", 
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        {"genre": "documentary", "year": 2019}
    ),
]

# Upsert Data
print("Upserting data")
index.upsert(data)
print("Process complete\n")

# Check on Insertions
index_stats = index.describe_index_stats()
print(f"Describe index stats => {index_stats}")





