import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key loaded: {api_key[:8]}...")

llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
response = llm.invoke("Say hello in one sentence.")
print(f"\nLLM Response: {response.content}")