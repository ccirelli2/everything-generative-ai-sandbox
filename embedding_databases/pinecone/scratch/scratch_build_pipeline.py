"""
Build a Pipeline to:
- Load text
- Create Embeddings
- Upload to pinecone database
- Query Embeddings from Database

Commands
=========================
- index.fetch(["id-3"])


References
=========================
- Query: https://docs.pinecone.io/docs/query-data
- https://huggingface.co/sentence-transformers
- name-spaces: https://docs.pinecone.io/docs/namespaces
"""

# Import libraries
import os
import logging
import pinecone
import itertools
from more_itertools import batched
from pinecone.core.client.model.vector import Vector
from decouple import config
from sentence_transformers import SentenceTransformer

# Directories
ROOT_DIR = "/home/christopher-cirelli/repositories/generative-ai-sandbox"
os.chdir(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'embedding_databases/pinecone/data/')

# Package Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


# Globals
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_FILENAME = "moby_dick.txt"
PINECONE_API_KEY = config('PINECONE_API_KEY')
PINECONE_ENV_NAME = config('PINECONE_ENV_NAME')

# Instantiate connection to pinecone
logger.info("Instantiating connection to pinecone\n")
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV_NAME)

# Load Sentences
logger.info("Loading sentences\n")
sentences = open(os.path.join(DATA_DIR, TEXT_FILENAME), 'r').readlines()

########################################################################################################################
# Create Embeddings
########################################################################################################################
'''
Instantiate model using model name.
Pass sentences to model. 
'''
logger.info("Instantiating embeddings model")
model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME)
logger.info(f"Embedding {len(sentences)} sentences")
embeddings = model.encode(sentences)
embeddingDim = len(embeddings[0])
logger.info(f"Embedding Dimensions => {embeddingDim}")
logger.info("Embedding complete\n")


########################################################################################################################
# Create Pinecone Index & Upsert Embeddings
########################################################################################################################

# Get Existing Indexes
pinecone_indexes = pinecone.list_indexes()
pinecone_collections = pinecone.list_collections()
logger.info(f"Pinecone Indexes => {pinecone_indexes} | Collections => {pinecone_collections}\n")

# Create Index
indexName = "moby-dick"
indexDim = embeddingDim
indexMetric = 'cosine'




# Structure Vectors


vector_inputs = create_vectors_with_metadata(
    sentences=sentences,
    embeddings=embeddings,
    author="moby-dick"
)


def upsert_data_batches(
        index: pinecone.Index,
        embeddings: list,
        batch: bool = False,
        batch_size: int = 100
) -> None:
    """

    :param index:
    :param embeddings:
    :param batch:
    :param batch_size:
    :return:
    """
    num_batches = int(len(embeddings) / batch_size)
    logger.info(f"Create {num_batches} batches of {batch_size} for upsert")
    batches = batched(embeddings, num_batches)

    count = 0
    for b in batches:
        logger.info(f"Upsert batch number => {count}")
        index.upsert(vectors=b)
        count += 1
    logger.info("Upsert completed")
    return None


upsert_data_batches(
    index=index,
    embeddings=vector_inputs,
    batch=True,
    batch_size=100
)


########################################################################################################################
# Similarity Search
########################################################################################################################

# Get Index Stats
"""
index_stats = index.describe_index_stats()
print(f"Describe index stats => {index_stats}")
"""

def query_database(
        index: pinecone.Index,
        sentence: str,
        model_name: str,
        num_sim: int
):
    """

    :param model_name:
    :param index:
    :param sentence:
    :param num_sim:
    :return:
    """
    logger.info(f"Query database for vectors similar to => {sentence}")
    logger.info("Creating sentence embedding")
    model = SentenceTransformer(model_name_or_path=model_name)
    embedding = model.encode(sentence).tolist()
    logger.info("Querying database")
    response = index.query(queries=[embedding], top_k=num_sim, include_metadata=True)
    logger.info("Returning most similar vector(s)")
    return response

"""
similar_sentences = query_database(
    index=index,
    sentence="Call me Ishmael",
    model_name=EMBEDDING_MODEL_NAME,
    num_sim=1
)

print(similar_sentences)
"""












