"""
Pipeline of functions to create embeddings, upsert and query db.
"""
# Import libraries
import logging
import pinecone
from sentence_transformers import SentenceTransformer

# Setup
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class EmbeddingsPipeline:
    def __init__(self):
        self.index=list
        self.text_obj=dict
        self.embeddings=list
        self.index_name=""
        self.index_dim=int
        self.index_metric=str

    def loadTextFromPath(self, path: str, author: str, title: str, source: str):
        """
        Load text from path and return sentences.

        :param path:
        :return:
        """
        logger.info(f"Loading text from path => {path}")
        sentences = open(path, 'r').readlines()
        assert isinstance(sentences, list)
        self.text_obj = {
            'author': author,
            'title': title,
            'source': source,
            'sentences': sentences
        }
        logger.info(f"Text loaded.  Number of sentences => {len(self.text_obj['sentences'])}\n")
        return self

    def createIndex(
            self,
            index_name: str,
            index_dim: int,
            index_metric: str,
            delete_duplicate_index: bool = False
    ) -> pinecone.Index:
        """
        :param index_name:
        :param index_dim:
        :param index_metric:
        :param delete_duplicate_index:
        :return:
        """
        logger.info(
            f"Request to creating a new index for => {index_name}, with dim => {index_dim} and metric => {index_metric}"
        )
        # Get List of existing indexes
        pinecone_indexes = pinecone.list_indexes()

        if index_name in pinecone_indexes:
            logger.info(f"Index {index_name} already exists.")

            if not delete_duplicate_index:
                logger.info("Loading existing index => {index_name} from pinecone")
                index = pinecone.Index(index_name=index_name)
                logger.info("Index load completed\n")
            else:
                logger.info(f"Option elected to delete existing index {index_name} and create anew")
                pinecone.delete_index(index_name=index_name)
                logger.info("Index deleted. Creating New Index")
                index = pinecone.create_index(
                    name=index_name,
                    dimension=index_dim,
                    metric=index_metric
                )
                logger.info(f"Index creation completed.\n")
        else:
            logger.info(f"No index found for index name => {index_name}")
            logger.info(f"Creating Pinecone index w/ Name {index_name} | Dim {index_dim} | Metric {index_metric}")
            index = pinecone.create_index(
                name=index_name,
                dimension=index_dim,
                metric=index_metric
            )
            logger.info("Index creation completed.  Existing indexes => {pinecone.list_indexes()}\n")

        # Return Index
        self.index = index
        return None

    def createEmbeddings(self, model_name_or_path: str):
        """
        Create embeddings from sentences.

        :param model_name_or_path:
        :return:
        """
        logger.info("Instantiating embeddings model")
        model = SentenceTransformer(model_name_or_path=model_name_or_path)
        logger.info(f"Embedding {len(self.text_obj['sentences'])} sentences")
        embeddings = model.encode(self.text_obj['sentences'])
        embeddingDim = len(embeddings[0])
        assert embeddingDim == self.
        logger.info(f"Embedding Dimensions => {embeddingDim}")
        logger.info("Embedding complete\n")
        return embeddings


    def createVectors(self):
        """

        :rtype: object
        :param author:
        :param embeddings:
        :param sentences:
        :return:
        """
        # Expose text parameters
        author = self.text_obj['author']
        title = self.text_obj['title']
        source = self.text_obj['source']
        sentences = self.text_obj['sentences']
        # author
        logger.info(f"Creating upsert vectors from author => {author}, title => {title}, source => {source}")

        # Results
        vectors = []
        num_sentences = len(sentences)
        num_embeddings = len(embeddings)
        assert num_sentences == num_embeddings,\
            f"Number of Sentences {num_sentences} != Number of Embeddings {num_embeddings}"
        # Combine Sentences & Embeddings
        sent_embed_vec = zip(embeddings, sentences)

        # Create Input Vectors
        count = 0
        for i, j in sent_embed_vec:

            # Create Vector using picone Vector class
            v = Vector(
                id=f"vector_{count}",
                values=i.tolist(),
                metadata={
                    "author": author,
                    "title":
                    "sentence": j}
            )
            vectors.append(v)
            count += 1

        logger.info("Process finished\n")
        # Return vectors
        return vectors



