"""
Script to show how to insert data into a collection.
"""
###################################################################################################
# Notes
# ===========================
# - You can insert data via client or use MilvusDM, an open source tool for importing and exporting
# - data to Milvus.
#
#
# References
# ===========================
# - https://milvus.io/docs/v2.0.x/insert_data.md
# - Work with string datatypes
#   - https://medium.com/vector-database/how-to-use-string-data-to-empower-your-similarity
#   search-applications-2a1ab906ca76
#   - Allows user to skip having to convert string to embedded vector when inserting data into
#    Milvus.
###################################################################################################

# Import Libraries
import os
from pymilvus import CollectionSchema, FieldSchema, DataType, connections, utility, Collection


# Directories
DIR_ROOT = "/home/oem/repositories/generative-ai-sandbox/vector-databases"
DIR_DATA = os.path.join(DIR_ROOT, 'data')

# Globals
FILE_NAME = "moby_dick.txt"

# Create a Connection
connections.connect("default", host="localhost", port="19530")

###################################################################################################
# Create Fields
###################################################################################################
book_id = FieldSchema(
    name="book_id",
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=True,
    description="Primary Key"
)

book_sentences = FieldSchema(
    name="sentence_name",
    dtype=DataType.FLOAT_VECTOR,
    max_length=100,
    dim=100
)


###################################################################################################
# Create Schema
###################################################################################################
schema = CollectionSchema(
    fields=[book_id, book_sentences],
    description="Test book search"
)

# Declare Collection Name
collection_name = "moby_dict_sentences"


###################################################################################################
# Create Collection
###################################################################################################
collection_exists = utility.has_collection(collection_name)

collection = Collection(
    name=collection_name,
    schema=schema,
    using="default",
    shard_num=2,
    consistency_level="Strong"
)

###################################################################################################
# Prepare Text Input
###################################################################################################

# Load Raw Text & Read All Lines
text_raw: list[str] = open(os.path.join(DIR_DATA, FILE_NAME), 'r').readlines()

def create_text_vector(text: str, chunk_size: int = 100) -> list:
    '''
    Function to create a list of text chunks of a specified size.

    :param text: raw text.
    :param chunk_size: size of chunk.
    :return: return list of text chunks.
    '''
    text_clean = " ".join(text_raw).replace("\n", " ").replace("  ", " ")
    text_chunks = [text_clean[i:i+chunk_size] for i in range(0, len(text_clean), chunk_size)]
    text_chunks = [x for x in text_chunks if len(x) == chunk_size]
    return text_chunks

# Create Text Chunks
text_chunks = create_text_vector(text_raw, chunk_size=100)

###################################################################################################
# Create Input Vector(s)
###################################################################################################
book_id_vector = [i for i in range(len(text_chunks))]




