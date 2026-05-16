# RAG Agent with LangChain — Built from Scratch

A step-by-step implementation of a RAG Agent using 
LangChain, OpenAI GPT-4o, and FAISS vector store.

## What this project does
- Loads a PDF document
- Chunks it intelligently  
- Creates embeddings using OpenAI
- Stores and searches using FAISS
- Answers questions using GPT-4o
- Uses a ReAct Agent with 4 custom tools

## Agent Tools
| Tool | Purpose |
|------|---------|
| search_document | RAG search over PDF |
| absence_calculator | Attendance/activity metrics |
| date_tool | Date calculations |
| compare_sections | Compare two topics side by side |

## Tech Stack
- Python 3.9
- LangChain
- OpenAI GPT-4o
- FAISS Vector Store
- ReAct Agent Pattern

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/rag-agent-langchain.git
cd rag-agent-langchain
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=your-key-here" > .env
python3 step9_10_agent.py
```

## Steps Covered
| File | What it does |
|------|-------------|
| step2_test.py | Test LLM connection |
| step3_load_pdf.py | Load PDF with PyPDFLoader |
| step4_chunking.py | Chunk with RecursiveCharacterTextSplitter |
| step5_embeddings.py | OpenAI embeddings + similarity test |
| step6_faiss.py | Build + save FAISS vector store |
| step7_retrieval.py | 3 types of retrieval |
| step8_rag_answer.py | RAG Q&A chain with sources |
| step9_10_agent.py | Full ReAct Agent with 4 tools |

## Author
Anees Fatima