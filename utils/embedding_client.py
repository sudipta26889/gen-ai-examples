import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import requests
import numpy as np
from typing import List


# Load environment variables first
load_dotenv()

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
LM_STUDIO_EMBEDDING_MODEL = os.getenv("LM_STUDIO_EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")

embedding_model = OpenAIEmbeddings(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY,
    model=LM_STUDIO_EMBEDDING_MODEL,
    check_embedding_ctx_length=False
)


def get_lm_studio_embeddings(texts: List[str], model_name: str = "text-embedding-nomic-embed-text-v1.5",
                             base_url: str = "http://127.0.0.1:1234") -> np.ndarray:
    """
    Get embeddings from LM Studio API for a list of texts.

    Args:
        texts: List of strings to embed
        model_name: Name of the embedding model in LM Studio
        base_url: Base URL of LM Studio API

    Returns:
        numpy array of embeddings with shape (n_texts, embedding_dim)
    """
    embeddings = []
    
    for text in texts:
        response = requests.post(
            f"{base_url}/v1/embeddings",
            headers={"Content-Type": "application/json"},
            json={
                "model": model_name,
                "input": text
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            embedding = result["data"][0]["embedding"]
            embeddings.append(embedding)
        else:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    
    return np.array(embeddings)


def get_lm_studio_embeddings_batch(texts: List[str], model_name: str = "text-embedding-nomic-embed-text-v1.5",
                                   base_url: str = "http://127.0.0.1:1234") -> np.ndarray:
    """
    Get embeddings from LM Studio API for a batch of texts.
    """
    response = requests.post(
        f"{base_url}/v1/embeddings",
        headers={"Content-Type": "application/json"},
        json={
            "model": model_name,
            "input": texts  # Try sending all texts at once
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        embeddings = [item["embedding"] for item in result["data"]]
        return np.array(embeddings)
    else:
        # Fallback to individual requests if batch doesn't work
        return get_lm_studio_embeddings(texts, model_name, base_url)
