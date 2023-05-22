"""

Reference:
=====================
PyPDFLoader(file_path: str)
Loads a PDF with pypdf and chunks at character level.
Loader also stores page numbers in metadatas.
"""
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

DIR_DATA = r"C:\Users\ccirelli\OneDrive - American International Group, Inc\Desktop\GitHub\generative-ai-sandbox\langchain_sandbox\data"
FILE_NAME = "portfolio-select-application-for-public-companies-brochure.pdf"
EMBEDDING_MODEL = EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

textSplitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
loader = PyPDFLoader(os.path.join(DIR_DATA, FILE_NAME))
#embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME)
pages = loader.load_and_split(text_splitter=textSplitter)
print(type(pages[0]))

# Create Embeddings
