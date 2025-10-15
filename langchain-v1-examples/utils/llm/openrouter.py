import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

openrouter_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)