import os
import time
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from .embedding_client import embedding_model

# Load environment variables first
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "test-key")

our_qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


def get_vector_store(collection_name="my-fav-indian-food"):
    """
    Get vector store instance for a collection.
    Only call this after ensuring the collection exists.
    """
    return QdrantVectorStore(
        client=our_qdrant_client,
        collection_name=collection_name,
        embedding=embedding_model,
    )


def add_documents_with_retry(vector_store, docs, batch_size=10, max_retries=3):
    """
    Add documents to vector store with batching and retry logic
    """
    total_docs = len(docs)
    successful_batches = 0
    
    for i in range(0, total_docs, batch_size):
        batch = docs[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_docs + batch_size - 1) // batch_size
        
        # print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
        
        for attempt in range(max_retries):
            try:
                vector_store.add_documents(batch)
                successful_batches += 1
                # print(f"✓ Batch {batch_num} added successfully")
                break
            
            except Exception as e:
                if "timeout" in str(e).lower() or "ResponseHandlingException" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Exponential backoff
                        print(
                            f"⚠ Timeout error in batch {batch_num}, attempt {attempt + 1}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"✗ Failed to add batch {batch_num} after {max_retries} attempts: {e}")
                        # Continue with next batch instead of stopping entirely
                else:
                    print(f"✗ Unexpected error in batch {batch_num}: {e}")
                    break
        
        # Small delay between batches to avoid overwhelming the server
        if i + batch_size < total_docs:
            time.sleep(1)