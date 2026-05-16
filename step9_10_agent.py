import os
import json
import math
import re
from datetime import date, datetime
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ════════════════════════════════════════════════
# LOAD FAISS (shared across all tools)
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
print("✅ FAISS loaded!\n")


# ════════════════════════════════════════════════
# STEP 9 — BUILD THE 4 TOOLS
# ════════════════════════════════════════════════

# ────────────────────────────────────────────────
# TOOL 1: Search Document (RAG)
# ────────────────────────────────────────────────
def search_document(query: str) -> str:
    """Search the marketing book for relevant content."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant content found in the document."

    results = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        content = re.sub(r'^\d+\n', '', doc.page_content.strip())
        results.append(f"[Page {page}]: {content[:300]}")

    return "\n\n".join(results)


tool_search = Tool(
    name="search_document",
    func=search_document,
    description="""Use this to search the marketing book 
    for information about any marketing topic.
    Input: a question or topic as plain text.
    Example input: 'what is content marketing?'"""
)


# ────────────────────────────────────────────────
# TOOL 2: Absence / Percentage Calculator
# ────────────────────────────────────────────────
def absence_calculator(input_str: str) -> str:
    """Calculate absence or activity percentage metrics."""
    try:
        params = json.loads(input_str)
    except json.JSONDecodeError:
        nums = re.findall(r'\d+\.?\d*', input_str)
        if len(nums) >= 2:
            params = {
                "absences": float(nums[0]),
                "total_days": float(nums[1])
            }
        else:
            return (
                "Could not parse input. "
                "Use JSON: {\"total_days\": 180, \"absences\": 20}"
            )

    total = float(params.get("total_days", 0))
    absent = float(params.get("absences", 0))
    threshold = float(params.get("threshold", 0.10))

    if total <= 0:
        return "Error: total_days must be greater than 0."

    attendance_rate = (total - absent) / total
    absence_rate = absent / total
    is_chronic = absence_rate >= threshold
    max_allowed = math.floor(threshold * total)
    days_left = max(0, max_allowed - int(absent))

    if absence_rate < 0.05:
        severity = "Satisfactory (< 5% absent)"
    elif absence_rate < 0.10:
        severity = "At-Risk (5-10% absent)"
    elif absence_rate < 0.20:
        severity = "Chronically Absent (10-20%)"
    else:
        severity = "Severely Chronically Absent (>=20%)"

    return (
        f"\n=== Absence / Activity Report ===\n"
        f"Total days           : {int(total)}\n"
        f"Days absent/inactive : {int(absent)}\n"
        f"Attendance rate      : {attendance_rate:.1%}\n"
        f"Absence rate         : {absence_rate:.1%}\n"
        f"Chronic absence flag : {'YES' if is_chronic else 'NO'}\n"
        f"Severity             : {severity}\n"
        f"Days left before threshold: {days_left}\n"
    )


tool_calculator = Tool(
    name="absence_calculator",
    func=absence_calculator,
    description="""Use this to calculate attendance, absence,
    or activity percentage metrics.
    Input: JSON with total_days and absences.
    Example input: {"total_days": 180, "absences": 22}"""
)


# ────────────────────────────────────────────────
# TOOL 3: Date Tool
# ────────────────────────────────────────────────
def date_tool(input_str: str) -> str:
    """Return today's date and compute date calculations."""
    today = date.today()

    month = today.month
    if 8 <= month <= 12:
        semester = "Fall Semester"
    elif 1 <= month <= 5:
        semester = "Spring Semester"
    else:
        semester = "Summer"

    result = [
        f"Today's date : {today.isoformat()}",
        f"Day of week  : {today.strftime('%A')}",
        f"Semester     : {semester}",
    ]

    input_str = input_str.strip()
    if input_str and input_str != "today":
        try:
            params = json.loads(input_str)
            if "start_date" in params:
                start = datetime.strptime(
                    params["start_date"], "%Y-%m-%d"
                ).date()
                delta = (today - start).days
                result.append(f"Days since {start}: {delta} days")
            if "end_date" in params:
                end = datetime.strptime(
                    params["end_date"], "%Y-%m-%d"
                ).date()
                delta = (end - today).days
                label = "remaining" if delta >= 0 else "overdue"
                result.append(
                    f"Days until {end}: {abs(delta)} days {label}"
                )
        except (json.JSONDecodeError, ValueError):
            try:
                start = datetime.strptime(
                    input_str, "%Y-%m-%d"
                ).date()
                delta = (today - start).days
                result.append(f"Days since {start}: {delta} days")
            except ValueError:
                result.append(f"Could not parse: '{input_str}'")

    return "\n".join(result)


tool_date = Tool(
    name="date_tool",
    func=date_tool,
    description="""Use this to get today's date or calculate
    days between dates.
    Input options:
    - "today" for just today's date
    - "2024-01-01" for days since that date
    - JSON: {"start_date": "2024-01-01", "end_date": "2024-06-30"}"""
)


# ────────────────────────────────────────────────
# TOOL 4: Compare Two Sections
# ────────────────────────────────────────────────
def compare_sections(input_str: str) -> str:
    """Compare two marketing topics from the book."""
    try:
        params = json.loads(input_str)
        topic_a = params.get("topic_a", "")
        topic_b = params.get("topic_b", "")
    except json.JSONDecodeError:
        if "|" in input_str:
            parts = input_str.split("|", 1)
            topic_a = parts[0].strip()
            topic_b = parts[1].strip()
        else:
            return (
                "Could not parse input. "
                "Use: {\"topic_a\": \"...\", \"topic_b\": \"...\"} "
                "or 'Topic A | Topic B'"
            )

    if not topic_a or not topic_b:
        return "Both topic_a and topic_b must be provided."

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    def fetch(query):
        docs = retriever.invoke(query)
        if not docs:
            return "No content found."
        results = []
        for d in docs:
            page = d.metadata.get('page', '?')
            content = re.sub(
                r'^\d+\n', '', d.page_content.strip()
            )[:200]
            results.append(f"[Page {page}]: {content}")
        return "\n".join(results)

    content_a = fetch(topic_a)
    content_b = fetch(topic_b)

    divider = "=" * 50
    return (
        f"\n{divider}\n"
        f"TOPIC A: '{topic_a}'\n"
        f"{divider}\n"
        f"{content_a}\n\n"
        f"{divider}\n"
        f"TOPIC B: '{topic_b}'\n"
        f"{divider}\n"
        f"{content_b}\n\n"
        f"Use the above to compare these two topics.\n"
    )


tool_compare = Tool(
    name="compare_sections",
    func=compare_sections,
    description="""Use this to compare two marketing topics
    from the book side by side.
    Input: JSON with topic_a and topic_b.
    Example: {"topic_a": "influencer marketing",
              "topic_b": "content marketing"}
    Or plain text: 'influencer marketing | content marketing'"""
)


# ════════════════════════════════════════════════
# STEP 10 — BUILD THE AGENT
# ════════════════════════════════════════════════

print("Building Agent...")

# ── All 4 tools ───────────────────────────────────
tools = [
    tool_search,
    tool_calculator,
    tool_date,
    tool_compare
]

# ── LLM ──────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=api_key
)

# ── ReAct Prompt ──────────────────────────────────
react_prompt = PromptTemplate.from_template("""
You are an intelligent assistant with access to a 
marketing book called "The New Marketing" and 
analytical tools.

Think step by step. Use the right tool for each question.
Always cite page numbers when referencing the book.
Context in the book may start with a page number — 
ignore that and focus on the actual content after it.

Available tools:
{tools}

Use this EXACT format:

Question: the input question
Thought: think about what to do
Action: tool name (must be one of [{tool_names}])
Action Input: input to the tool
Observation: result from the tool
... (repeat Thought/Action/Observation if needed)
Thought: I now have enough to answer
Final Answer: your complete answer

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")

# ── Create Agent ──────────────────────────────────
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

# ── Create Agent Executor ─────────────────────────
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,               # shows ALL thinking!
    max_iterations=6,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)
print("✅ Agent ready!\n")


# ════════════════════════════════════════════════
# TEST THE AGENT WITH 4 QUESTIONS
# ════════════════════════════════════════════════

questions = [
    # Tool 1 — RAG search
    "What is content marketing in simple terms?",

    # Tool 2 — calculator
    "A marketing campaign ran for 90 days but was active only 45 days. What percentage was it active?",

    # Tool 3 — date
    "What is today's date and how many days since Jan 1 2025?",

    # Tool 4 — compare two topics
    "Compare influencer marketing with personal branding from the book",
]

for question in questions:
    print("\n" + "=" * 55)
    print(f"QUESTION: {question}")
    print("=" * 55)

    result = agent_executor.invoke({"input": question})

    print(f"\n✅ FINAL ANSWER:\n{result['output']}")
    steps = result.get("intermediate_steps", [])
    print(f"\n[{len(steps)} tool call(s) made]")
    print("=" * 55)
