"""
Notes
=============================


References
=============================
1. Using Pre-trained models: https://huggingface.co/learn/nlp-course/chapter4/2?fw=pt
2. Installing libraries: https://huggingface.co/docs/transformers/installation
3. GP4 Chatbot: https://github.com/mayooear/gpt4-pdf-chatbot-langchain
4. Quickstart: https://huggingface.co/docs/huggingface_hub/quick-start
5. Hub-library: https://huggingface.co/docs/huggingface_hub/index
6. Quick Tour: https://huggingface.co/docs/transformers/quicktour
7. Huggingface Course: https://huggingface.co/learn/nlp-course/chapter1/1


"""

# Load Libraries
from transformers import pipeline

# Load a Model
cambert_fill_mask = pipeline("fill-mask", model="camembert-base")


