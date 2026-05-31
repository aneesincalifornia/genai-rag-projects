# app/embeddings.py

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List
import os

def create_vector_store(chunks: List[Document]):
    """
    Converts text chunks into embeddings (vectors)
    and stores them in a FAISS vector store.

    What happens here:
    1. Each chunk is sent to OpenAI's embedding model
    2. OpenAI returns a vector (list of 1536 numbers) for each chunk
    3. FAISS stores all these vectors so we can search them later

    Think of it as building a searchable index of meaning,
    not just keywords.
    """

    print("Creating embeddings... this may take a moment.")

    # OpenAI's text-embedding-3-small is fast and cheap
    # Each chunk gets converted to 1536 numbers
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # FAISS takes all chunks + embeddings and builds a vector index
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    print(f"Vector store created with {vector_store.index.ntotal} vectors")

    return vector_store


def save_vector_store(vector_store, path: str = "faiss_index"):
    """
    Saves the vector store to disk so we don't have to
    re-embed every time we restart the app.
    Embedding costs money — we only want to do it once.
    """
    vector_store.save_local(path)
    print(f"Vector store saved to: {path}")


def load_vector_store(path: str = "faiss_index"):
    """
    Loads a previously saved vector store from disk.
    Use this on app restarts instead of re-embedding.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    vector_store = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print(f"Vector store loaded from: {path}")
    return vector_store