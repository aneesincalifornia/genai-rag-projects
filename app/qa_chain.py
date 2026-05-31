# app/qa_chain.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

def build_qa_chain(retriever):
    """
    Builds the full QA chain using LangChain's modern LCEL syntax.

    Flow:
    1. User asks a question
    2. Retriever finds top 4 relevant chunks
    3. Chunks + question are formatted into a prompt
    4. GPT reads the prompt and returns a grounded answer

    The prompt tells GPT to ONLY use the provided context.
    This is the main guardrail against hallucination.
    """

    prompt_template = """
    You are a helpful HR assistant. Use ONLY the following context
    to answer the question. If the answer is not in the context,
    say "I don't have enough information in this document to answer that."
    Do not make up answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    def format_docs(docs):
        """Joins all retrieved chunks into one string for the prompt"""
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain: modern LangChain way of wiring components
    # question goes to retriever AND directly to prompt
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("QA chain ready")
    return chain