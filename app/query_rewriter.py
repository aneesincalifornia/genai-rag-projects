# app/query_rewriter.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

def build_query_rewriter():
    """
    Query rewriting fixes the vocabulary mismatch problem in RAG.

    Problem:
    User asks  : "How many days of leave am I entitled to?"
    Document says : "12 workweeks of FMLA leave"

    Different words, same meaning.
    Semantic search struggles when vocabulary is too different.

    Solution:
    Before retrieval, ask the LLM to rephrase the user's question
    into language closer to what a formal HR/legal document would use.

    This dramatically improves retrieval accuracy.
    """

    prompt_template = """
    You are an expert at rephrasing questions to improve 
    document retrieval accuracy.
    
    Rephrase the following question using formal, precise language
    that would match how an HR policy or legal document is written.
    
    Keep the meaning exactly the same but use professional 
    HR and legal terminology.
    
    Return ONLY the rephrased question. Nothing else.
    No explanation. No preamble. Just the rephrased question.

    Original question: {question}
    
    Rephrased question:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question"]
    )

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    rewriter = prompt | llm | StrOutputParser()

    return rewriter


def rewrite_query(rewriter, question: str) -> str:
    """
    Takes the user's original question and returns
    a rewritten version optimized for document retrieval.
    """
    rewritten = rewriter.invoke({"question": question})
    print(f"\nOriginal question  : {question}")
    print(f"Rewritten question : {rewritten}")
    return rewritten