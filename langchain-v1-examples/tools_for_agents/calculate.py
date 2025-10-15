from langchain.tools import tool
from ddgs import DDGS
from langgraph.config import get_stream_writer

@tool
def calculate(x: str) -> str:
    """Useful for doing math calculations."""
    writer = get_stream_writer()
    writer(f"Calculating: {x}")
    result = str(eval(x))
    writer(f"Result: {result}")
    return result
