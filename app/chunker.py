# app/chunker.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def chunk_documents(pages: List[Document]):
    """
    Takes a list of pages and splits them into smaller chunks.

    Why we chunk:
    - LLMs have a context window limit
    - We only want to send RELEVANT chunks to the LLM
    - Smaller chunks = more precise retrieval

    Strategy: RecursiveCharacterTextSplitter
    - Tries to split at paragraphs first
    - Then sentences
    - Then words
    - Never cuts arbitrarily mid-meaning

    chunk_size    = 1000 characters per chunk
    chunk_overlap = 200 characters repeated between chunks
                    so meaning is not lost at boundaries
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(pages)

    print(f"Total chunks created: {len(chunks)}")
    print(f"Example chunk:\n{'-'*40}")
    print(chunks[0].page_content)
    print(f"{'-'*40}")

    return chunks