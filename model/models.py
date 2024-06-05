import os
from langchain_openai import ChatOpenAI





#import the model
def llm():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    return llm