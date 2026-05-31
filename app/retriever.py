# app/retriever.py

from langchain_community.vectorstores import FAISS

def get_retriever(vector_store: FAISS):
    """
    Creates a retriever from the vector store.

    The retriever takes a question, converts it to an embedding,
    and finds the top K most similar chunks.

    k=4 means we fetch the 4 most relevant chunks.
    Why 4? Enough context for a good answer,
    not so many that we overwhelm the LLM with noise.
    """

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    print("Retriever ready")
    return retriever