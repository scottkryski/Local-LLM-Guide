import torch
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient, models

# --- 1. Load the Embedding Model ---
# Use a specific model path
model_path = "ibm-granite/granite-embedding-30m-english"


# Load the model and tokenizer from Hugging Face
print("---Loading embedding model---")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode
print("Embedding model loaded.")
print("")

# --- 2. Initialize Qdrant Client ---
client = QdrantClient(path="./qdrant_db")
print("---Initializing Qdrant Client---")
print("Qdrant client initialized.")
print("")

# --- 3. Create the Qdrant Collection ---
vector_size = 384  # Matching the granite-30m-english model
collection_name = "example_collection"

# Check if collection already exists, delete for this example
if client.collection_exists(collection_name):
    print(f"---Deleting collection: {collection_name}---")
    client.delete_collection(collection_name)

# Use create_collection to ensure a fresh start
print(f"---Creating collection: {collection_name}---")
client.create_collection(
    collection_name = collection_name,
    vectors_config = models.VectorParams(
        size = vector_size, 
        distance = models.Distance.COSINE
    ),
)
print(f"Collection '{collection_name}' created with vector size {vector_size}.")
print("")

# --- 4. Prepare Documents ---
documents = [
    {"id": 1, "content": "The IBM Granite models are a family of open-source language models."},
    {"id": 2, "content": "Qdrant is a high-performance vector database written in Rust."},
    {"id": 3, "content": "Ollama allows you to run large language models locally on your own machine."},
    {"id": 4, "content": "A RAG system combines a retriever with a generator to answer questions."},
]

# --- 5. Generate Embeddings and Upsert into Qdrant ---
points_to_upsert = []
with torch.no_grad(): # Disable gradient calculation for efficiency
    print("---Generating embeddings and upserting into Qdrant---")
    for doc in documents:
        # Tokenize the document content
        tokenized_input = tokenizer(doc["content"], padding=True, truncation=True, return_tensors='pt')
        
        # Get the model's output
        model_output = model(**tokenized_input)
        
        # Perform CLS pooling (as recommended for this model)
        embedding = model_output.last_hidden_state[:, 0]
        
        # Normalize the embedding
        normalized_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        # Create the Qdrant point
        point = models.PointStruct(
            id=doc["id"],
            vector=normalized_embedding.tolist(), # Convert to list
            payload={"text": doc["content"]} # Store the original text
        )
        points_to_upsert.append(point)

# Upsert all points to the collection in one go
client.upsert(
    collection_name=collection_name,
    points=points_to_upsert,
    wait=True  # Wait for the operation to complete
)

print(f"Successfully upserted {len(documents)} documents into Qdrant.")
print("")

# --- 6. Verify the upload ---
count_result = client.count(collection_name=collection_name, exact=True)
print("---Verifying the upload---")
print(f"Number of documents in collection: {count_result.count}")
print("")
