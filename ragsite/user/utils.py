import os
from urllib.request import urlretrieve
import numpy as np
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from django.conf import settings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pickle


def generate_vectors():
    # Load pdf files in the local directory
    loader = PyPDFDirectoryLoader(settings.FILE_STORAGE_DIR)

    docs_before_split = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50,
    )
    docs_after_split = text_splitter.split_documents(docs_before_split)

    avg_doc_length = lambda docs: sum([len(doc.page_content) for doc in docs]) // len(docs)
    avg_char_before_split = avg_doc_length(docs_before_split)
    avg_char_after_split = avg_doc_length(docs_after_split)

    huggingface_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        # alternatively use "sentence-transformers/all-MiniLM-l6-v2" for a light and faster experience.
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = FAISS.from_documents(docs_after_split, huggingface_embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    with open(settings.VECTOR_DB_FILE_NAME, "wb") as file:
        pickle.dump(retriever, file)
    return retriever


def ask_query(query):
    with open(settings.VECTOR_DB_FILE_NAME, "rb") as file:
        # Load the object from the file
        retriever = pickle.load(file)
    hf = HuggingFacePipeline.from_model_id(
        # model_id="mistralai/Mistral-7B-v0.1",
        model_id="Mistral-7B-v0.1",
        task="text-generation",
        pipeline_kwargs={"temperature": 0, "max_new_tokens": 300}
    )
    llm = hf
    llm.invoke(query)
    retrievalQA = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    result = retrievalQA.invoke({"query": query})
    return result
