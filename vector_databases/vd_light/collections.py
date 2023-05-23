"""

"""
import logging
from sentence_transformers import SentenceTransformer
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def create_embeddings(sentences: list[str], embedding_model_name: str, index_dim: int):
    """
    Create embeddings from sentences.

    :return:
    """
    logger.info("Instantiating embeddings model")
    model = SentenceTransformer(model_name_or_path=embedding_model_name)
    logger.info(f"Embedding {len(sentences)} sentences")
    embeddings = model.encode(sentences)
    embedding_dim_obs = len(embeddings[0])
    assert embedding_dim_obs == index_dim, "Embedding dimensions do not match index dimensions"
    logger.info(f"Embedding Dimensions => {embedding_dim_obs}")
    logger.info("Embedding complete\n")
    return embeddings
