import nest_asyncio
nest_asyncio.apply()

import os
import json
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# RAGAS imports
# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from datasets import Dataset

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ════════════════════════════════════════════════
# STEP 1 — Load our existing RAG pipeline
# ════════════════════════════════════════════════

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

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=api_key
)

prompt_template = """
You are a helpful assistant answering questions about
a marketing book called "The New Marketing".

Use ONLY the context below to answer the question.
The context may start with a page number — ignore that
and focus on the actual content after it.

If the answer is not in the context say
"I could not find this in the document."

Always mention which page you found the answer on.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)
print("✅ RAG chain ready!\n")

# ════════════════════════════════════════════════
# STEP 2 — Create test dataset
# ════════════════════════════════════════════════

# These are our test questions WITH ground truth answers
# Ground truth = what the correct answer SHOULD be
test_cases = [
    {
        "question": "What is content marketing?",
        "ground_truth": "Content marketing is about sharing helpful information rather than pitching. It is a dialogue not a monologue and focuses on interacting with buyers and helping solve their problems authentically."
    },
    {
        "question": "What are the 7 steps to building a personal brand?",
        "ground_truth": "The 7 steps are: 1. Do an environmental scan 2. Create a brand value proposition 3. Position your brand 4. Figure out your brand story 5. Develop a content strategy 6. Develop a content distribution strategy 7. Measure results."
    },
    {
        "question": "How do influencers help brands grow?",
        "ground_truth": "Influencers help brands grow by scaling word of mouth marketing through social media. Their followers trust them like friends making recommendations more effective than traditional advertising."
    },
    {
        "question": "What is a buyer persona?",
        "ground_truth": "A buyer persona is a data driven representation of your ideal customer based on real data and research. It helps brands understand customer needs behaviors and motivations to create more targeted marketing."
    },
    {
        "question": "What is the role of data in modern marketing?",
        "ground_truth": "Data plays a central role in modern marketing by enabling real time insights into consumer behavior allowing brands to personalize the customer journey and make evidence based decisions rather than relying on intuition."
    },
]

# ════════════════════════════════════════════════
# STEP 3 — Run agent and collect results
# ════════════════════════════════════════════════

print("Running RAG pipeline on test questions...")
print("=" * 55)

questions      = []
answers        = []
contexts       = []
ground_truths  = []

for i, test in enumerate(test_cases, 1):
    print(f"\nQuestion {i}: {test['question']}")

    # Run the RAG chain
    result = rag_chain.invoke({"query": test["question"]})

    # Collect the answer
    answer = result["result"]
    print(f"Answer: {answer[:150]}...")

    # Collect retrieved chunks as context
    source_docs = result["source_documents"]
    context_list = [doc.page_content for doc in source_docs]

    # Print context precision insight
    print(f"Chunks retrieved: {len(context_list)}")
    for j, doc in enumerate(source_docs, 1):
        print(f"  Chunk {j} | Page {doc.metadata.get('page','?')}")

    # Store everything
    questions.append(test["question"])
    answers.append(answer)
    contexts.append(context_list)
    ground_truths.append(test["ground_truth"])

print("\n" + "=" * 55)
print("✅ All questions answered!\n")

# ════════════════════════════════════════════════
# STEP 4 — Build RAGAS dataset
# ════════════════════════════════════════════════

print("Building RAGAS evaluation dataset...")

ragas_dataset = Dataset.from_dict({
    "question"    : questions,
    "answer"      : answers,
    "contexts"    : contexts,
    "ground_truth": ground_truths,
})

print(f"✅ Dataset ready: {len(ragas_dataset)} test cases\n")

# ════════════════════════════════════════════════
# STEP 5 — Run RAGAS evaluation
# ════════════════════════════════════════════════

print("Running RAGAS evaluation...")
print("(This calls OpenAI API to score each metric)")
print("Please wait...\n")

results = evaluate(
    dataset=ragas_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=llm,
    embeddings=embedding_model,
)

# ════════════════════════════════════════════════
# STEP 6 — Display results
# ════════════════════════════════════════════════

print("\n" + "=" * 55)
print("   RAGAS EVALUATION RESULTS")
print("=" * 55)

scores = results.to_pandas()

# Overall scores
print("\n📊 OVERALL SCORES:")
print(f"  Faithfulness      : {results['faithfulness']:.3f}")
print(f"  Answer Relevancy  : {results['answer_relevancy']:.3f}")
print(f"  Context Precision : {results['context_precision']:.3f}")
print(f"  Context Recall    : {results['context_recall']:.3f}")

# Score interpretation
print("\n📋 WHAT THIS MEANS:")
metrics = {
    "Faithfulness"      : results["faithfulness"],
    "Answer Relevancy"  : results["answer_relevancy"],
    "Context Precision" : results["context_precision"],
    "Context Recall"    : results["context_recall"],
}

for metric, score in metrics.items():
    if score >= 0.8:
        status = "Excellent ✅"
    elif score >= 0.7:
        status = "Good ✅"
    elif score >= 0.5:
        status = "Needs improvement ⚠️"
    else:
        status = "Poor - fix urgently ❌"
    print(f"  {metric:<22}: {score:.3f} → {status}")

# Per question breakdown
print("\n📝 PER QUESTION BREAKDOWN:")
print("-" * 55)
for i, row in scores.iterrows():
    print(f"\nQ{i+1}: {questions[i][:50]}...")
    print(f"  Faithfulness      : {row.get('faithfulness', 'N/A')}")
    print(f"  Answer Relevancy  : {row.get('answer_relevancy', 'N/A')}")
    print(f"  Context Precision : {row.get('context_precision', 'N/A')}")
    print(f"  Context Recall    : {row.get('context_recall', 'N/A')}")

# Save results to JSON
output = {
    "overall": {
        "faithfulness"      : round(results["faithfulness"], 3),
        "answer_relevancy"  : round(results["answer_relevancy"], 3),
        "context_precision" : round(results["context_precision"], 3),
        "context_recall"    : round(results["context_recall"], 3),
    },
    "per_question": []
}

for i, row in scores.iterrows():
    output["per_question"].append({
        "question"          : questions[i],
        "answer"            : answers[i],
        "faithfulness"      : round(float(row.get("faithfulness", 0)), 3),
        "answer_relevancy"  : round(float(row.get("answer_relevancy", 0)), 3),
        "context_precision" : round(float(row.get("context_precision", 0)), 3),
        "context_recall"    : round(float(row.get("context_recall", 0)), 3),
    })

with open("evaluation_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n✅ Results saved to evaluation_results.json")
print("=" * 55)