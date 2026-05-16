import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# ── Step 1: Load the PDF ──────────────────────────────
PDF_PATH = "The New Marketing - Cheryl Burgess.pdf"

print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
print(f"✅ Pages loaded: {len(pages)}")

# ── Step 2: Chunk it ──────────────────────────────────
print("\nChunking the PDF...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 800,   # max characters per chunk
    chunk_overlap = 100,   # overlap between chunks
    separators    = ["\n\n", "\n", ".", " "]  # where to split
)

chunks = splitter.split_documents(pages)

# ── Step 3: Inspect the chunks ────────────────────────
print(f"✅ Total chunks created: {len(chunks)}")
print(f"   Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")

print(f"\n--- Chunk 1 ---")
print(f"Content : {chunks[0].page_content}")
print(f"Metadata: {chunks[0].metadata}")

print(f"\n--- Chunk 2 ---")
print(f"Content : {chunks[1].page_content}")
print(f"Metadata: {chunks[1].metadata}")

print(f"\n--- Chunk 10 ---")
print(f"Content : {chunks[9].page_content}")
print(f"Metadata: {chunks[9].metadata}")