"""
Scratch file to learn how to convert documemnts (sentences, paragraphs, etc.) into vectors.
"""
############################################################################################
# Tutorial Content
# ===========================
# - Review the relevant models: bag-of-words, Word2Vec, Doc2Vec
# - Load and preprocess the training and test corpora (see Corpus)
# - Train a Doc2Vec Model model using the training corpus
# - Demonstrate how the trained model can be used to infer a Vector
# - Assess the model
# - Test the model on the test corpus
#
# Model(s) Review
# ===========================
# - Bag-of-Words: converts sentences into a fixed length vector of integers based on the count
#   that the word appears in the document.
# - Word2Vec:  embeds words in a lower dimension vector space using a shallow neural network.
# - Paragraph Vector (Doc2Vec): embeds paragraphs into a vector space.
#   PV-DM is analogous to Word2Vec CBOW. The doc-vectors are obtained by training a neural
#   network on the synthetic task of predicting a center word based an average of both context
#   word-vectors and the full document’s doc-vector.

# References
# ===========================
# - Tutorial: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html
############################################################################################

############################################################################################
# Setup
############################################################################################
import os
import io
import re
import gensim
from sklearn.metrics.pairwise import cosine_similarity


# Globals
DIR_ROOT = "/home/oem/repositories/generative-ai-sandbox/gensim-sandbox"
DIR_DATA = os.path.join(DIR_ROOT, 'data')

# Import Data
textRaw = open(os.path.join(DIR_DATA, "moby_dick.txt"))


############################################################################################
# Preprocess Text
############################################################################################
def preprocess_text(text_raw, tokens_only=False):
    """
    Clean and tokenize text

    :param tokens_only:
    :param text_raw:
    :return:
    """
    lines = text_raw.readlines()
    for i, line in enumerate(lines):
        tokens = gensim.utils.simple_preprocess(line)
        if tokens_only:
            yield tokens
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

trainCorpus = list(preprocess_text(text_raw=textRaw))
testCorpus = list(preprocess_text(text_raw=textRaw, tokens_only=True))


############################################################################################
# Train Model
############################################################################################

# Instantiate Model
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

# Create Vocabulary
'''
get vocabulary = print(model.wv.key_to_index)
get vocab attr = print(model.wv.get_vecattr('which', 'count'))
'''
model.build_vocab(trainCorpus)

# Train Model
model.train(trainCorpus, total_examples=model.corpus_count, epochs=model.epochs)


# Get Cosine Similarity between predicted vector and actual vector
yhat = model.infer_vector(trainCorpus[1].words).reshape(1, -1)
actual = model.docvecs[1].reshape(1, -1)
similarity = round(cosine_similarity(yhat, actual)[0][0], 3)

print(f'Similarity => {similarity}')

