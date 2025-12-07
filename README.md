# PDF Q&A with LangChain (RAG over Documents)

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that can read any PDF and answer questions about its content using **LangChain** and **OpenAI**.

## What it does

- Loads a PDF (e.g., a resume, report, or policy).
- Splits pages into overlapping text chunks using `RecursiveCharacterTextSplitter`.
- Generates embeddings with `text-embedding-3-small`.
- Stores embeddings + chunks in an in-memory **FAISS** vector store.
- Uses a `RetrievalQA` chain with `gpt-4o` to answer questions.
- Shows both the answer and the source chunks for transparency.

## Project structure

```text
pdf-qa-langchain/
├─ main.py            # Entry point: build index + interactive Q&A
├─ requirements.txt   # Python dependencies
├─ .env.example       # Environment variable template
└─ sample_docs/
   └─ sample.pdf      # (optional) Example PDF for testing
