import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# ── Step 1: Load & Chunk (same as before) ─────────────
print("Loading and chunking PDF...")
loader = PyPDFLoader("The New Marketing - Cheryl Burgess.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = splitter.split_documents(pages)
print(f"✅ Chunks ready: {len(chunks)}")

# ── Step 2: Create Embedding Model ────────────────────
print("\nSetting up embedding model...")
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",  # OpenAI's embedding model
    openai_api_key=api_key
)
print("✅ Embedding model ready")

# ── Step 3: Embed a single sentence (to understand it) ──
print("\n--- Embedding a single sentence ---")
sample_text = "content marketing strategy for brands"
single_embedding = embedding_model.embed_query(sample_text)

print(f"Text    : '{sample_text}'")
print(f"Numbers : {single_embedding[:5]}...")  # first 5 numbers
print(f"Total dimensions: {len(single_embedding)}")

# ── Step 4: Embed two similar sentences ───────────────
print("\n--- Comparing similar vs different sentences ---")

text_a = "content marketing helps brands grow"
text_b = "creating valuable content attracts customers"
text_c = "I enjoy eating pizza on weekends"

emb_a = embedding_model.embed_query(text_a)
emb_b = embedding_model.embed_query(text_b)
emb_c = embedding_model.embed_query(text_c)

# Calculate similarity (dot product)
import numpy as np

sim_ab = np.dot(emb_a, emb_b)  # similar meaning
sim_ac = np.dot(emb_a, emb_c)  # different meaning

print(f"\nText A: '{text_a}'")
print(f"Text B: '{text_b}'  ← similar meaning")
print(f"Text C: '{text_c}'  ← different meaning")
print(f"\nSimilarity A↔B : {sim_ab:.4f}  (higher = more similar)")
print(f"Similarity A↔C : {sim_ac:.4f}  (should be lower)")

if sim_ab > sim_ac:
    print("\n✅ Embeddings work! Similar texts scored higher than different texts.")
else:
    print("\n⚠️ Unexpected result - check your API key")

# ── Step 5: Show what happens to our chunks ────────────
print("\n--- Embedding our PDF chunks ---")
print(f"We have {len(chunks)} chunks to embed")
print("Each chunk will become 1536 numbers")
print(f"Total numbers to store: {len(chunks) * 1536:,}")
print("\n(We'll actually store them in Step 6 — FAISS)")