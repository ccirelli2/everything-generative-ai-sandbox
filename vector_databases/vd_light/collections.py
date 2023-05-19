"""

"""
import logging
from sentence_transformers import SentenceTransformer
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def create_embeddings(sentences: list[text], embedding_model_name: str):
    """
    Create embeddings from sentences.

    :return:
    """
    logger.info("Instantiating embeddings model")
    model = SentenceTransformer(model_name_or_path=embedding_model_name)
    logger.info(f"Embedding {len(sentences)} sentences")
    embeddings = model.encode(sentences)
    embedding_dim_obs = len(embeddings[0])
    assert embedding_dim_obs == self.index_dim, "Embedding dimensions do not match index dimensions"
    logger.info(f"Embedding Dimensions => {embedding_dim_obs}")
    logger.info("Embedding complete\n")
    return return embedding