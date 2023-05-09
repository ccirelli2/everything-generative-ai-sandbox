"""
Pipeline of functions to create embeddings, upsert and query db.
"""
# Import libraries
import os
import logging
import pinecone
from more_itertools import batched
from sentence_transformers import SentenceTransformer
from pinecone.core.client.model.vector import Vector


# Setup
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class EmbeddingsPipeline:
    def __init__(
            self,
            path: str,
            file_name: str,
            author: str,
            title: str,
            source: str,
            embedding_model_name: str
    ):
        """

        :param path:
        :param file_name:
        :param author:
        :param title:
        :param source:

        :param index:
        :param text_obj: dict that includes sentences and metadata associated with the source of the test.
        :param embeddings: sentence embeddings.

        """
        self.path2file=path
        self.file_name=file_name
        self.author=author
        self.title=title
        self.source=source
        self.embedding_model_name=embedding_model_name
        self.index=list
        self.sentences=str
        self.embeddings=list
        self.index_name=str
        self.index_dim=int
        self.index_metric=str

    def pipeline(self):
        """

        :return:
        """
        self.load_text()
        self.create_index()


    def load_text(self):
        """
        Load text from file.
        """
        logger.info(f"Loading text from path => {self.path}")
        sentences = open(os.path.join(self.path, self.file_name), 'r').readlines()
        assert isinstance(sentences, list), 'Sentences must be a list'
        self.sentences = sentences
        logger.info(f"Text loaded.  Number of sentences => {len(self.sentences)}\n")
        return self

    def create_index(
            self,
            delete_duplicate_index: bool = False
    ) -> pinecone.Index:
        """

        :param delete_duplicate_index:
        :return:
        """
        logger.info(
            f"Request to creating a new index for => {self.index_name}, with dim => {self.index_dim} and metric => {self.index_metric}"
        )
        # Get List of existing indexes
        pinecone_indexes = pinecone.list_indexes()

        if self.index_name in pinecone_indexes:
            logger.info(f"Index {self.index_name} already exists.")

            if not delete_duplicate_index:
                logger.info("Loading existing index => {index_name} from pinecone")
                index = pinecone.Index(index_name=self.index_name)
                logger.info("Index load completed\n")
            else:
                logger.info(f"Option elected to delete existing index {self.index_name} and create anew")
                pinecone.delete_index(index_name=self.index_name)
                logger.info("Index deleted. Creating New Index")
                index = pinecone.create_index(
                    name=self.index_name,
                    dimension=self.index_dim,
                    metric=self.index_metric
                )
                logger.info(f"Index creation completed.\n")
        else:
            logger.info(f"No index found for index name => {self.index_name}")
            logger.info(f"Creating Pinecone index w/ Name {self.index_name} | Dim {self.index_dim} | Metric {self.index_metric}")
            index = pinecone.create_index(
                name=self.index_name,
                dimension=self.index_dim,
                metric=self.index_metric
            )
            logger.info(f"Index creation completed.  Existing indexes => {pinecone.list_indexes()}\n")

        # Return Index
        self.index = index
        return self

    def create_embeddings(self):
        """
        Create embeddings from sentences.

        :return:
        """
        logger.info("Instantiating embeddings model")
        model = SentenceTransformer(model_name_or_path=self.embedding_model_name)
        logger.info(f"Embedding {len(self.sentences)} sentences")
        embeddings = model.encode(self.sentences)
        embedding_dim_obs = len(embeddings[0])
        assert embedding_dim_obs == self.index_dim, "Embedding dimensions do not match index dimensions"
        self.embeddings = embeddings
        logger.info(f"Embedding Dimensions => {embedding_dim_obs}")
        logger.info("Embedding complete\n")
        return self

    def create_vectors(self):
        """

        :rtype: object
        :return:
        """
        logger.info(f"Creating upsert vectors from author => {self.author}, title => {self.title}, source => {self.source}")

        # Results
        vectors = []
        num_sentences = len(self.sentences)
        num_embeddings = len(self.embeddings)
        assert num_sentences == num_embeddings,\
            f"Number of Sentences {num_sentences} != Number of Embeddings {num_embeddings}"

        # Create Input Vectors
        count = 0
        for i, j in zip(self.embeddings, self.sentences):

            # Create Vector using picone Vector class
            v = Vector(
                id=f"vector_{count}",
                values=i.tolist(),
                metadata={
                    "author": self.author,
                    "title": self.title,
                    "source": self.source,
                    "sentence": j}
            )
            vectors.append(v)
            count += 1

        logger.info("Process finished\n")
        # Return vectors
        return vectors
