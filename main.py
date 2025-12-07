import os
import sys

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


def load_api_key():
    """
    Load OPENAI_API_KEY from .env or environment.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Set it in .env or environment variables."
        )
    return api_key


def build_vectorstore_from_pdf(pdf_path: str) -> FAISS:
    """
    1) Load the PDF
    2) Chunk it
    3) Embed each chunk
    4) Build a FAISS vector store
    """

    print(f"\n[INFO] Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print(f"[INFO] Loaded {len(docs)} pages.")

    # --- Chunking (important for RAG) ---
    # We use recursive splitter:
    # - chunk_size ~ 800 tokens (or chars)
    # - chunk_overlap keeps some context between chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    split_docs = text_splitter.split_documents(docs)
    print(f"[INFO] Split into {len(split_docs)} chunks.")

    # --- Embeddings ---
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # --- Vector store (FAISS in-memory) ---
    print("[INFO] Building FAISS vector store...")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    print("[INFO] Vector store ready.")

    return vectorstore


def build_qa_chain(vectorstore: FAISS) -> RetrievalQA:
    """
    Build a RetrievalQA chain:
    - uses FAISS retriever
    - uses GPT-4o as LLM
    """

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # retrieve top 3 relevant chunks
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,  # deterministic answers
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",  # simplest: stuff context into prompt
    )

    return qa_chain


def interactive_qa(qa_chain: RetrievalQA):
    """
    Simple CLI loop to ask questions about the PDF.
    """
    print("\n[INFO] You can now ask questions about the PDF.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("❓ Your question: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("\n[INFO] Exiting. Goodbye!")
            break

        if not query:
            continue

        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        source_docs = result["source_documents"]

        print("\n💡 Answer:")
        print(answer)

        print("\n📚 Top source chunks (for transparency):")
        for i, doc in enumerate(source_docs, start=1):
            print(f"\n--- Source {i} ---")
            # Show only first 400 chars of each chunk
            print(doc.page_content[:400].strip(), "...")
        print("\n" + "-" * 80 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF file not found: {pdf_path}")
        sys.exit(1)

    # Load key (fails early if missing)
    load_api_key()

    # Build vector store and QA chain
    vectorstore = build_vectorstore_from_pdf(pdf_path)
    qa_chain = build_qa_chain(vectorstore)

    # Start interactive Q&A loop
    interactive_qa(qa_chain)


if __name__ == "__main__":
    main()
