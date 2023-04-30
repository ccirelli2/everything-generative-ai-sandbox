"""
References
==========
1. https://docs.langchain.com/docs/quickstart
"""
import os
from pprint import pprint
from langchain.llms import OpenAI

################################################################################
# Instantiate OpenAI Model & Ask Questions
################################################################################
llm = OpenAI(model_name="text-davinci-003", n=2, best_of=2, temperature=0.9)
#response = llm("Tell me a joke")
#response = llm("What is two plus two?")
#response = llm("What is the first derivative of x squared?")
#response = llm("What does American International Group (AIG) the insurance company do?")
#response = llm("Please list the risk factors found in the AIG 10-K from 2020.")
#
pprint(response)
