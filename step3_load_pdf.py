import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# ── Load the PDF ──────────────────────────────────────
PDF_PATH = "The New Marketing - Cheryl Burgess.pdf"

print(f"Loading PDF: {PDF_PATH} ...")
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

# ── Inspect what we got ───────────────────────────────
print(f"\n✅ Total pages loaded: {len(pages)}")
print(f"\n--- Page 1 Preview ---")
print(f"Page number : {pages[0].metadata['page']}")
print(f"Source      : {pages[0].metadata['source']}")
print(f"\nFirst 500 characters of text:\n")
print(pages[0].page_content[:500])

print(f"\n--- Page 2 Preview ---")
print(pages[1].page_content[:500])