"""
This file is used to create a collection in Milvus.
"""
################################################################################
# First step is to create a collection, which is similar to a table in a RDBMS.
#
# Collection
# - A collection in Milvus is equivalent to a table in a relational database
# - management system (RDBMS).
# - In Milvus, collections are used to store and manage entities.
#
# Entities
# - An entity consists of a group of fields that represent real world objects.
# Each entity in Milvus is represented by a
# - unique primary key.
#
# Fields
# - Fields are the units that make up entities. Fields can be structured data
# (e.g., numbers, strings) or vectors.
#
# References
# ===================
# 1. https://milvus.io/docs/v2.0.x/create_collection.md
################################################################################

# Import Libraries
import os
from pprint import pprint
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, utility, connections

# Create a Connection
connections.connect("default", host="localhost", port="19530")

# Create Fields
book_id = FieldSchema(
    name="book_id",
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=True,
    description="Primary Key"
)
word_count = FieldSchema(
    name="work_count",
    dtype=DataType.INT64
)
book_intro = FieldSchema(
    name="book_intro",
    dtype=DataType.FLOAT_VECTOR,
    dim=2
)

# Create Collection Schema
schema = CollectionSchema(
    fields=[book_id, word_count, book_intro],
    description="Test book search"
)

# Name Collection
collection_name = "book_search"


# Create Collection
collection_exists = utility.has_collection(collection_name)

if not collection_exists:
    print(f"Collection does not exist.  Creating collection => {collection_name}")
    collection = Collection(
        name=collection_name,
        schema=schema,
        using="default",
        shard_num=2,
        consistency_level="Strong"
    )

else:
    print("Collection exists")

# Print Collections
#existing_collections = utility.list_collections()

