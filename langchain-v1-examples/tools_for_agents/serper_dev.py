import http.client
import json
import os
from dotenv import load_dotenv  

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def web_search_using_serper(query: str) -> str:
    """Web search using serper.dev. Search the internet based on given query"""
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
        "q": query,
    })
    headers = {
    'X-API-KEY': SERPER_API_KEY,
    'Content-Type': 'application/json'
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")
