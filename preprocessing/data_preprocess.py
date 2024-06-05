import os

from langchain import hub
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory



# Load the Files

file_paths = ["./data/paper_1.pdf", "./data/paper_2.pdf"]


def preprocessing(file_paths,chunk_size=1000,chunk_overlap=200):
    all_pages = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        all_pages.extend(pages)

    # Splitting the text

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    splits=text_splitter.split_documents(all_pages)

    # Saving in vector store

    vectors_storer=Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings())

    # Retreiving the documents

    retreive=vectors_storer.as_retriever()

    return retreive


def contextualize_q_system_prompt_fun():
    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is.")
    return contextualize_q_system_prompt


def system_prompt():
    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}")
    return system_prompt