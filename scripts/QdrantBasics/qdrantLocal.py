from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_db")

# print the list of collections
print(client.get_collections())