import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ── Step 1: Load FAISS from disk (FREE - no API call) ──
print("Loading FAISS index from disk...")
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
)
vectorstore = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)
print("✅ Loaded from disk instantly!\n")

# ── Step 2: Basic Retrieval ────────────────────────────
print("=" * 50)
print("RETRIEVAL TYPE 1: Basic Similarity Search")
print("=" * 50)

query = "how do influencers help in marketing?"
results = vectorstore.similarity_search(query, k=4)

print(f"Query: '{query}'")
print(f"Found {len(results)} chunks:\n")
for i, doc in enumerate(results, 1):
    print(f"  Chunk {i} | Page {doc.metadata['page']}")
    print(f"  {doc.page_content[:150]}...")
    print()

# ── Step 3: Retrieval WITH Scores ─────────────────────
print("=" * 50)
print("RETRIEVAL TYPE 2: With Similarity Scores")
print("=" * 50)

query2 = "what is personal branding?"
results_with_scores = vectorstore.similarity_search_with_score(query2, k=4)

print(f"Query: '{query2}'\n")
for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"  Chunk {i} | Page {doc.metadata['page']} | Score: {score:.4f}")
    print(f"  {doc.page_content[:150]}...")
    print()

# ── Step 4: Retriever Object (used in agents!) ─────────
print("=" * 50)
print("RETRIEVAL TYPE 3: Retriever Object (Agent-ready)")
print("=" * 50)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

query3 = "how does social media affect customer engagement?"
docs = retriever.invoke(query3)

print(f"Query: '{query3}'")
print(f"Found {len(docs)} chunks:\n")
for i, doc in enumerate(docs, 1):
    print(f"  Chunk {i} | Page {doc.metadata['page']}")
    print(f"  {doc.page_content[:150]}...")
    print()

# ── Step 5: Understanding scores ──────────────────────
print("=" * 50)
print("UNDERSTANDING SCORES")
print("=" * 50)
print("""
In FAISS similarity scores:
  Lower score  = MORE similar  (it's a distance!)
  Higher score = LESS similar

  Score < 0.3  → Very relevant  ✅
  Score 0.3-0.5 → Relevant      ✅  
  Score > 0.7  → Less relevant  ⚠️
""")

# Show best vs worst result
best = results_with_scores[0]
worst = results_with_scores[-1]
print(f"  Best match  → Page {best[0].metadata['page']} | Score: {best[1]:.4f} ✅")
print(f"  Worst match → Page {worst[0].metadata['page']} | Score: {worst[1]:.4f}")