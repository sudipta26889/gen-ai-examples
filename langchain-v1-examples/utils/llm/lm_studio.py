import os
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

lmstudio_llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    temperature=0.1,
    api_key="no-key-needed-for-lm-studio",
    base_url=os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
    max_tokens=120000,
    streaming=True,
    verbose=True,
)