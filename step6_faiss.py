import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ── Step 1: Load & Chunk ──────────────────────────────
print("Loading and chunking PDF...")
loader = PyPDFLoader("The New Marketing - Cheryl Burgess.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = splitter.split_documents(pages)
print(f"✅ Chunks ready: {len(chunks)}")

# ── Step 2: Embedding Model ───────────────────────────
print("\nSetting up embedding model...")
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
)
print("✅ Embedding model ready")

# ── Step 3: Build FAISS Vector Store ─────────────────
print("\nBuilding FAISS vector store...")
print("(This calls OpenAI API to embed all 1106 chunks)")
print("Please wait — this is the only time we pay for this!...")

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)
print("✅ FAISS vector store built!")

# ── Step 4: Save to disk ──────────────────────────────
print("\nSaving vector store to disk...")
vectorstore.save_local("faiss_index")
print("✅ Saved! Folder 'faiss_index' created.")

# ── Step 5: Reload from disk (prove it works!) ────────
print("\nReloading from disk (no API call this time)...")
loaded_vectorstore = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)
print("✅ Reloaded successfully from disk!")

# ── Step 6: Quick test search ─────────────────────────
print("\n--- Quick Test Search ---")
query = "what is content marketing strategy?"
results = loaded_vectorstore.similarity_search(query, k=3)

print(f"Query: '{query}'")
print(f"Top {len(results)} results found:\n")

for i, doc in enumerate(results, 1):
    print(f"Result {i}:")
    print(f"  Page    : {doc.metadata['page']}")
    print(f"  Content : {doc.page_content[:200]}...")
    print()