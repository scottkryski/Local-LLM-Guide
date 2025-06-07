from qdrant_client import QdrantClient, models

# Connect to a local, on-disk Qdrant instance
client = QdrantClient(path = "./qdrant_db")

collection_name = "example_collection"
# Define the vector size from our model
vector_size = 384

# Delete all collections, purely for example
if client.collection_exists(collection_name):
    print(f"Deleting collection: {collection_name}")
    client.delete_collection(collection_name)

# Create the collection
# You can use client.create_collection to ensure you start fresh
print(f"Creating collection: {collection_name}")
client.create_collection(
    collection_name = collection_name,
    vectors_config = models.VectorParams(
        size = vector_size, 
        distance = models.Distance.COSINE
    ),
)

# print collection exists
print(f"Collection exists: {collection_name}")
print(client.collection_exists(collection_name))
