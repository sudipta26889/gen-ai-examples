import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.embedding_client import get_lm_studio_embeddings_batch, get_lm_studio_embeddings


def rerank_results(embedding_model, search_query, searched_df, ):
    # Check if DataFrame is empty
    if len(searched_df) == 0:
        print("Warning: No documents to rerank. Returning empty DataFrame.")
        searched_df['rerank_score'] = []
        return searched_df
    
    # Debug: Check input data
    print(f"Number of documents: {len(searched_df.page_content)}")
    print(f"Search query: '{search_query}'")
    
    new_doc_embeddings = get_lm_studio_embeddings_batch(
        searched_df.page_content.tolist(),
        model_name="text-embedding-nomic-embed-text-v1.5"
    )
    
    # Debug: Check document embeddings
    print(f"Document embeddings shape: {new_doc_embeddings.shape}")
    
    query_embedding = get_lm_studio_embeddings(
        [search_query],  # Pass as a list with single item
        model_name="text-embedding-nomic-embed-text-v1.5"
    )[0]
    
    # Debug: Check query embedding
    print(f"Query embedding shape: {query_embedding.shape}")
    
    # Ensure query_embedding is a numpy array
    query_embedding = np.array(query_embedding)
    
    similarity_scores = cosine_similarity(
        query_embedding.reshape(1, -1),
        new_doc_embeddings
    )
    searched_df['rerank_score'] = similarity_scores[0].tolist()
    return searched_df