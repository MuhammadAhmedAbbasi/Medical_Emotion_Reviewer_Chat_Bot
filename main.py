import os

from preprocessing import data_preprocess
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
import streamlit as st




file_paths = ["./data/paper_1.pdf", "./data/paper_2.pdf"]

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
retreive=data_preprocess.preprocessing(file_paths)
contextualize_q_system_prompt=data_preprocess.contextualize_q_system_prompt_fun()
system_prompt=data_preprocess.system_prompt()



contextualize_q_prompt=ChatPromptTemplate.from_messages(
   [ ("system",contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}"),
    ]
)


history_aware_retreiver=create_history_aware_retriever(llm,retreive,contextualize_q_prompt)

qa_prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}"),
    ]
)

stuff=create_stuff_documents_chain(llm,qa_prompt)

rag_chain=create_retrieval_chain(history_aware_retreiver,stuff)


store={}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain =RunnableWithMessageHistory(

    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

config={
    "configurable":{"session_id":"abc123"}
}

st.title('Medical Emotional Review Paper Helper ChatBot')
input_text=st.text_input("Search the topic you want")


# Check if there is an input text
if input_text:
    # Invoke the chain with the input text and session configuration
    response = conversational_rag_chain.invoke(
        {"input": input_text}, 
        config={"configurable": {"session_id": "abc123"}}
    )

    # Retrieve the 'answer' from the response and display it
    if 'answer' in response:
        st.write(response['answer'])
    else:
        st.write("No response generated.")