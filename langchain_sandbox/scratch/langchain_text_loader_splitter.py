"""
References
============================================
https://docs.trychroma.com/getting-started
https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/chroma.html
"""
import os
import sys
import numpy as np

import logging
from sentence_transformers import SentenceTransformer
from time import time
from uuid import uuid4
from pprint import pprint as pprint

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader


# Library Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# Globals
DIR_DATA = r"/vector_databases/vd_light/data"
FILE_NAME = "moby10b.txt"
EMBEDDING_MODEL = EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Load Text
"""
loader.load() returns a list of langchain.schema.Document objects.  Each document object has the following attributes:
- page_content
- metadata
"""
loader = TextLoader(os.path.join(DIR_DATA, FILE_NAME))
document = loader.load()  # returns a list [0] = langchain.schema.Document

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(document)  # returns a list of langchain.schemaDocument

print(docs[0].page_content.split(' '))
