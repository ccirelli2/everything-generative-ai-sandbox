"""
Pipeline of functions to create embeddings, upsert and query db.
"""
# Import libraries
import logging
from sentence_transformers import SentenceTransformer

# Setup
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


# Create a Function to Create Embeddings
class CreateEmbeddings:
    def __init__(
            self,
            sentences: list,
            transformer_name: str
    ):
        logger.info("Create Embedding class created")
        self.sentences = sentences
        self.transformer_name = transformer_name
        self.embeddings = None

    def embed(self):
        logger.info(f"Creating len({self.sentences}) embeddings")
        model = SentenceTransformer(self.transformer_name)
        self.embeddings = model.encode(self.sentences)
        logger.info(f"Embeddings Created")
        return self





