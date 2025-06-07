import torch
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient, models

# --- Stage 1: Setup and Initialization ---

print("--- Stage 1: Setup and Initialization ---")

# Define the model path for the Hugging Face model
model_path = "ibm-granite/granite-embedding-30m-english"

# Load the tokenizer and model
# This will download the model from Hugging Face on the first run
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode (important for inference)
    print("✅ Embedding model and tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model/tokenizer. Ensure 'transformers' and 'torch' are installed.")
    print(f"Error: {e}")
    exit()


# Initialize the Qdrant client to use a local, on-disk database
try:
    client = QdrantClient(path="./qdrant_db")
    print("✅ Qdrant client initialized.")
except Exception as e:
    print(f"❌ Failed to initialize Qdrant client. Ensure 'qdrant-client' is installed.")
    print(f"Error: {e}")
    exit()


# Define the collection configuration
vector_size = 384  # Dimension for the granite-30m-english model
collection_name = "example_collection"

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
print(f"✅ Qdrant collection '{collection_name}' created with vector size {vector_size}.")


# --- Stage 2: Document Population ---

print("\n--- Stage 2: Document Population ---")

# Define the documents to be stored
documents = [
    {"id": 1, "content": "The IBM Granite models are a family of open-source language models for enterprise use."},
    {"id": 2, "content": "Qdrant is a high-performance vector database and similarity search engine written in Rust."},
    {"id": 3, "content": "Ollama allows you to run large language models, like Llama 3, locally on your own machine."},
    {"id": 4, "content": "Retrieval-Augmented Generation (RAG) is an architecture that combines a retriever with a generator to answer questions."},
]
print(f"Found {len(documents)} documents to process.")

# Generate embeddings and prepare points for upsertion
points_to_upsert = []
with torch.no_grad():  # Disable gradient calculation for efficiency
    for doc in documents:
        # Tokenize the document content
        tokenized_input = tokenizer(doc["content"], padding=True, truncation=True, return_tensors='pt')
        
        # Get the model's output embeddings
        model_output = model(**tokenized_input)
        
        # Perform CLS pooling (takes the embedding of the [CLS] token)
        embedding = model_output.last_hidden_state[:, 0]
        
        # Normalize the embedding to a unit vector
        normalized_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        # Create the Qdrant point with vector and payload
        point = models.PointStruct(
            id=doc["id"],
            vector=normalized_embedding.tolist()[0],  # Convert tensor to list
            payload={"text": doc["content"]}         # Store the original text as metadata
        )
        points_to_upsert.append(point)

# Upsert all points to the collection
client.upsert(
    collection_name=collection_name,
    points=points_to_upsert,
    wait=True  # Wait for the operation to complete
)
print(f"✅ Successfully upserted {len(documents)} documents into Qdrant.")


# --- Stage 3: Retrieval (Searching) ---

print("\n--- Stage 3: Retrieval (Searching) ---")

# Define the user's search query
query_text = "How can I run a large model on my computer?"
print(f"Search query: '{query_text}'")

# Embed the query using the same model and process
with torch.no_grad():
    tokenized_query = tokenizer(query_text, padding=True, truncation=True, return_tensors='pt')
    model_output = model(**tokenized_query)
    query_embedding = model_output.last_hidden_state[:, 0]
    normalized_query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)

# Perform the vector search in Qdrant
search_results = client.search(
    collection_name=collection_name,
    query_vector=normalized_query_embedding.tolist()[0],
    limit=3  # Return the top 3 most similar results
)

# Print the results
print("\n✅ Search Results:")
if not search_results:
    print("No results found.")
else:
    for i, result in enumerate(search_results):
        print(f"  Result {i+1}:")
        print(f"    - ID:      {result.id}")
        print(f"    - Score:   {result.score:.4f}")  # Similarity score (higher is better for Cosine)
        print(f"    - Text:    {result.payload['text']}")
        print("-" * 40)