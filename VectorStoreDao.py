import os
from typing import List, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import TextLoader, PyPDFLoader, CSVLoader


class VectorStoreDao():
    def __init__(self, persist_directory: str = "./chroma_db", embedding_model: str = None):
        self.persist_directory = persist_directory
        self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=HuggingFaceEmbeddings())
