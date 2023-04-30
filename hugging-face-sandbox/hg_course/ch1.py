"""

Notes
=========
1. pipeline(): most basic object in the transformer library. It connects a model with its necessary preprocessing and
   postprocessing steps, allowing us to directly input any text and get an answer.

Examples of currently availble pipelines:
- feature-extraction (get the vector representation of a text)
- fill-mask
- ner (named entity recognition)
- question-answering
- sentiment-analysis
- summarization
- text-generation
- translation
- zero-shot-classification

2. Using Any Model From Hub in a Pipeline

References
==========
1. https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt
2. Available Pipelines https://huggingface.co/docs/transformers/main_classes/pipelines
3. Model Hub: https://huggingface.co/models

"""

from transformers import pipeline


## Pipeline Example for Sentiment Analysis
def get_sentiment(sentences: list) -> list:
    classifier = pipeline("sentiment-analysis")
    sentiment_scores = classifier(sentences)
    return sentiment_scores

#sentences = ["Today is a good day to code", "Today is a bad day to code"]
#sentiment = get_sentiment(sentences)


## Pipeline Example Named Entity Recognition
def get_named_entities(sentence: str) -> str:
    classifier = pipeline("ner")
    named_entities = classifier(sentence, aggregation_strategy="simple")
    return named_entities
#sentence = "Hugging Face Inc. is a company based in New York City."
#response = get_named_entities(sentence)

## Use Any Model From Hub in a Pipeline
"""
deepset/roberta-base-squad2
"""



'''
Please write a function in python that takes in a string and returns the same string.
'''
