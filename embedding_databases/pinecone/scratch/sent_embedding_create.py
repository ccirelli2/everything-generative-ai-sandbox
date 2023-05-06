"""
References
=========================
- https://huggingface.co/sentence-transformers
"""

# Import libraries
import logging
from sentence_transformers import SentenceTransformer

# Setup
logger = logging.getLogger(__name__)
logger.setLevel(level="INFO")

# Create a Function to Create Embeddings
class CreateEmbeddings:
    def __init__(
            self,
            sentences: list,
            transformer_name: str
        ):
        logger.info("Create Embedding class created")
        self.sentences=sentences
        self.transformer_name=transformer_name
        self.embeddings=None


    def ecode(self):
        logger.info(f"Creating len({self.sentences}) embeddings")
        model = SentenceTransformer(self.transformer_name)
        self.embeddings = model.ecode(self.sentences)
        logger.info(f"Embeddings Created")
        return None


if __name__ == "main":
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SENT = ["Today is a good day to code", "Yesterday was a good day to code"]
    embed = CreateEmbeddings(
            sentences=SENT,
            transformer_name=MODEL
            )

