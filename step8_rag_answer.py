import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ── Step 1: Load FAISS (free, from disk) ──────────────
print("Loading FAISS index...")
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
)
vectorstore = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
print("✅ FAISS loaded!\n")

# ── Step 2: Load LLM ──────────────────────────────────
print("Loading LLM...")
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=api_key
)
print("✅ LLM ready!\n")

# ── Step 3: Custom Prompt ─────────────────────────────
# This tells GPT-4o HOW to use the retrieved chunks
prompt_template = """
You are a helpful assistant answering questions about 
a marketing book called "The New Marketing".

Use the context below to answer the question.
The context may start with a page number — ignore that 
and focus on the actual content after it.

If you genuinely cannot find relevant information, 
say "I could not find this in the document."

Always mention which page you found the answer on.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ── Step 4: Build RAG Chain ───────────────────────────
print("Building RAG chain...")
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",        # stuff all chunks into one prompt
    retriever=retriever,
    return_source_documents=True,   # show which pages were used
    chain_type_kwargs={"prompt": prompt}
)
print("✅ RAG chain ready!\n")

# ── Step 5: Ask Questions! ────────────────────────────
questions = [
    "What are the 7 steps to building a personal brand?",
    "How do influencers help brands grow?",
    "What is content marketing in simple terms?",
]

for question in questions:
    print("=" * 55)
    print(f"❓ Question: {question}")
    print("=" * 55)

    result = rag_chain.invoke({"query": question})

    print(f"\n💬 Answer:\n{result['result']}")

    print(f"\n📄 Sources used:")
    for doc in result['source_documents']:
        print(f"   Page {doc.metadata['page']} → "
              f"{doc.page_content[:80]}...")
    print()