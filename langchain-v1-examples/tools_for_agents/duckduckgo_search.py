from langchain.tools import tool
from ddgs import DDGS
from langgraph.config import get_stream_writer

@tool
def duckduckgo_search(query: str) -> str:
    """Search for information in the web using DuckDuckGo."""
    writer = get_stream_writer()
    # stream any arbitrary data
    writer(f"Searching for: {query}")
    ddgs_results = DDGS().text(query, max_results=5)
    writer(f"Result From DuckDuckGo: {ddgs_results}")
    return ddgs_results
