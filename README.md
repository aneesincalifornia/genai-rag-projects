<<<<<<< HEAD
# HR Policy RAG Bot

I built this to understand how RAG actually works under the hood,
not just read about it but build it piece by piece and see where
it breaks.

The bot lets you ask plain English questions about any PDF document
and get answers grounded in the actual content, with page citations.

---

## What it does

- Loads a PDF document
- Cleans and pre-processes the text (headers, footers, broken lines)
- Splits text into smart overlapping chunks
- Converts chunks into vector embeddings using OpenAI
- Finds the most relevant chunks for your question using FAISS
- Answers your question using only the document content
- Cites the exact pages the answer came from

---

## Tech Stack

- Python 3.13
- LangChain
- OpenAI GPT-3.5-turbo + text-embedding-3-small
- FAISS (vector store)
- PyPDF

---

## Project Structure
```
books-rag/
├── app/
│   ├── pdf_loader.py    # Loads PDF pages
│   ├── cleaner.py       # Pre-processes and cleans text
│   ├── chunker.py       # Splits text into overlapping chunks
│   ├── embeddings.py    # Creates and stores vector embeddings
│   ├── retriever.py     # Searches for relevant chunks
│   └── qa_chain.py      # Connects retriever to LLM
├── data/                # Put your PDF files here
├── main.py              # Entry point, run this
├── .env                 # Your API key (never committed)
├── .gitignore
└── README.md
```

---

## How to run locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/books-rag.git
cd books-rag
```

**2. Create a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install langchain langchain-openai langchain-community \
            langchain-text-splitters pypdf faiss-cpu \
            python-dotenv openai
```

**4. Add your OpenAI API key**
```bash
# Create a .env file and add this one line:
OPENAI_API_KEY=your_key_here
```

**5. Add your PDF**
```bash
# Copy any PDF into the data/ folder
# Update the filename in main.py line 19 if needed
```

**6. Run**
```bash
python main.py
```

---

## Key concepts

**Why RAG?**
LLMs do not know your documents. RAG gives them context by
retrieving the most relevant chunks from your document and
grounding the answer in actual content rather than the model's
general training knowledge.

**Chunking strategy**
I used RecursiveCharacterTextSplitter with chunk_size=1000
and chunk_overlap=200. It tries to split at paragraph boundaries
first, then sentences, then words. The overlap of 200 characters
makes sure sentences that fall at chunk boundaries are not lost.
I chose this over fixed size chunking because policy documents
have inconsistent paragraph lengths and fixed cuts destroy meaning.

**Why pre-processing matters**
Raw PDFs are noisy. Headers, footers, page numbers, and broken
line breaks all end up in your chunks if you skip this step.
I treat pre-processing in RAG the same way I treat EDA before
building an ML model. Garbage in, garbage out applies here too.

**Hallucination prevention**
The prompt explicitly tells the LLM to answer only from the
provided context and respond with "I don't have enough information"
if the answer is not there. This keeps the bot honest.

---

## Sample output
```
Your question: Who is eligible for FMLA leave?

Answer: Employees who have worked for their employer for at least
12 months, have worked at least 1250 hours in the last 12 months,
and work at a location where the employer has at least 50 employees
within 75 miles of their worksite are eligible for FMLA leave.

Sources:
  Page 3: If you work for a covered employer, you need to meet
          additional criteria to be eligible to take FMLA...
  Page 4: Your employer is not covered...
```

---

## What I learned building this

- Chunking strategy matters more than the LLM you pick
- Raw PDFs are noisy and pre-processing is not optional
- Question phrasing affects retrieval quality. If your words
  do not match the document vocabulary, retrieval fails even
  when the answer is there
- The prompt template is your main guardrail against hallucination
- Chunk overlap is underrated. Without it you lose meaning at
  boundaries and get incomplete answers
- LangChain refactored heavily in version 0.2 and imports moved
  to separate packages. Good to know before you hit those errors
  in production

---

## What I would improve next

- Add a query rewriting step so user questions are rephrased
  to match document vocabulary before retrieval
- Try semantic chunking for documents where topics shift frequently
- Build a Streamlit UI so non-technical users can interact with it
- Add support for multiple PDFs at once
- Cache the vector store so the app restarts without re-embedding
- Add evaluation metrics to measure retrieval quality

---

## Author

Anees Fatima
Senior Data Scientist and AI/GenAI Engineer
[LinkedIn](https://www.linkedin.com/in/anees-fatima-inamadar/)
=======
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

**Environment setup**
conda create -n llms python=3.11
conda activate llms
pip install -r requirements.txt
>>>>>>> 31ce4850e0ab21112b1a75f9d1ebe68bf82d1fcb
