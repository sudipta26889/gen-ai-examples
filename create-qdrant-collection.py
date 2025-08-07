from qdrant_client.http.models import VectorParams, Distance
from utils.vectordb_client import our_qdrant_client, get_vector_store

# First, create the collection
our_qdrant_client.create_collection(
    collection_name="my-fav-indian-food",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

print("Collection 'my-fav-indian-food' created successfully!")

# Now you can safely get the vector store
vector_store = get_vector_store("my-fav-indian-food")
print("Vector store initialized successfully!")