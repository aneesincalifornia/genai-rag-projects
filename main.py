# main.py

from dotenv import load_dotenv
from app.pdf_loader import load_pdf
from app.cleaner import clean_documents, filter_short_chunks, remove_duplicate_chunks
from app.chunker import chunk_documents
from app.embeddings import create_vector_store, save_vector_store
from app.retriever import get_retriever
from app.qa_chain import build_qa_chain
from app.query_rewriter import build_query_rewriter, rewrite_query
import os

# Load API key from .env file
load_dotenv()

def main():
    print("=" * 50)
    print("HR Policy RAG Bot with Query Rewriting")
    print("=" * 50)

    # Step 1: Load PDF
    pages = load_pdf("data/employeeguide_HRpolicy.pdf")

    # Step 2: Clean pages BEFORE chunking
    pages = clean_documents(pages)

    # Step 3: Chunk it
    chunks = chunk_documents(pages)

    # Step 4: Clean chunks AFTER chunking
    chunks = filter_short_chunks(chunks)
    chunks = remove_duplicate_chunks(chunks)

    # Step 5: Create embeddings and vector store
    vector_store = create_vector_store(chunks)

    # Step 6: Save vector store
    save_vector_store(vector_store)

    # Step 7: Build retriever
    retriever = get_retriever(vector_store)

    # Step 8: Build QA chain
    qa_chain = build_qa_chain(retriever)

    # Step 9: Build query rewriter
    rewriter = build_query_rewriter()

    # Step 10: Ask questions in a loop
    print("\nAsk me anything about the HR policy.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        # Rewrite the question before retrieval
        rewritten_question = rewrite_query(rewriter, question)

        # Get answer using rewritten question
        answer = qa_chain.invoke(rewritten_question)
        print(f"\nAnswer: {answer}")

        # Show sources using rewritten question
        print("\nSources:")
        source_docs = retriever.invoke(rewritten_question)
        for doc in source_docs:
            page = doc.metadata.get("page", "unknown")
            print(f"  Page {page}: {doc.page_content[:100]}...")

        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()