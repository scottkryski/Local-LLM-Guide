---
layout: default
title: "Part 3: Vector Stores with Qdrant"
---

### Introduction to Qdrant

Qdrant is a high-performance, open-source vector database that is well-suited for building RAG applications. It offers both a server-based version for production use and a lightweight local mode for development and smaller projects.

While many other vector databases exist each with their own positives and negatives, we will be using Qdrant going forward.

---

### Setting up Qdrant

You have two main options for running Qdrant: as a Docker container, or as a fully local, serverless instance.

**I'd personally recommend Option 2 for now, as its easier to get started with.**

#### Option 1: Running Qdrant as a Server with Docker

If you need to scale or have your vector store accessed by multiple services this is likely the way to go.

First make sure you have Docker installed:
https://www.docker.com/products/docker-desktop/

Then with Docker desktop running, put this command in your terminal:

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

This command starts a Qdrant container and persists its data in a `qdrant_storage` directory on your host machine.

#### Option 2: Running Qdrant Locally

For quick development, testing, or self-contained applications, you can run Qdrant directly in your Python script. It will store its data on disk without needing a separate server process. This is the simplest way to get started.

First install the client library: `pip install qdrant-client`

---

### Using Qdrant with Python

The process changes depending on which previous option you went with.

#### Connecting to the Docker Server

To connect to the server, you need to use its URL:

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
```

#### Connecting to the Local Server

If you're running it without the server, simply provide a path to a local directory where the database files will be stored.

```python
from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_db")
```

#### Creating a Collection

Ok, once you have your client, the rest of the code is the same for both options.

To store your vectors you need to create a "collection". The `vector_size` must match the output dimensions of your embedding model (Typically 384 or 768, some go up to 1024 or 1536).

```python
from qdrant_client import QdrantClient, models

# Connect to a local, on-disk Qdrant instance
client = QdrantClient(path="./qdrant_db")

# Define the vector size from our model
vector_size = 768

# Create the collection
# You can use client.recreate_collection to ensure you start fresh
client.recreate_collection(
    collection_name="my_enterprise_collection",
    vectors_config=models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE
    ),
)

print(f"Collection 'my_enterprise_collection' created successfully with vector size {vector_size}.")
```

For our use case, I'm leaning towards using the IBM Granite Embedding Models. They are commercially friendly under an Apache 2.0 License, and are optimized for retrieval.

While some Granite models may be available through Ollama, the most reliable method is to pull them directly from their source on Hugging Face.

This means you'll need to install the Hugging Face `transformers` library: `pip install transformers sentence-transformers`

> **Note:**
>
> This also requires you have PyTorch installed!

IBM provides 4 embedding models:

| Model                               | Dimensions |
| ----------------------------------- | ---------- |
| granite-embedding-30m-english       | 384        |
| granite-embedding-125m-english      | 768        |
| granite-embedding-107m-multilingual | 384        |
| granite-embedding-278m-multilingual | 768        |

### Example using IBM's Granite Embedding Models

From: https://www.ibm.com/granite/docs/models/embedding/

This code will load the IBM Granite Embedding model from Hugging Face, Initialize the Qdrant client, create a collection with the correct vector size (384), loop through a list of example documents, creating an embedding for each one, and then store the resulting vectors and document content in the collection.

#### Creating and Populating the Collection

```python
import torch
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient, models

# --- 1. Load the Embedding Model ---
model_path = "ibm-granite/granite-embedding-30m-english"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model.eval()
print("✅ Embedding model loaded.")

# --- 2. Initialize Qdrant Client ---
client = QdrantClient(path="./qdrant_db")
print("✅ Qdrant client initialized.")

# --- 3. Create the Qdrant Collection ---
vector_size = 384  # Matching the granite-30m-english model
collection_name = "example_collection" # Using the correct collection name

# Use recreate_collection to ensure a fresh start every time
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE
    ),
)
print(f"✅ Collection '{collection_name}' created.")

# --- 4. Prepare Documents ---
documents = [
    {"id": 1, "content": "The IBM Granite models are a family of open-source language models."},
    {"id": 2, "content": "Qdrant is a high-performance vector database written in Rust."},
    {"id": 3, "content": "Ollama allows you to run large language models locally on your own machine."},
    {"id": 4, "content": "A RAG system combines a retriever with a generator to answer questions."},
]

# --- 5. Generate Embeddings and Upsert into Qdrant ---
points_to_upsert = []
with torch.no_grad():
    for doc in documents:
        tokenized_input = tokenizer(doc["content"], padding=True, truncation=True, return_tensors='pt')
        model_output = model(**tokenized_input)
        embedding = model_output.last_hidden_state[:, 0]
        normalized_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        # Using 'page_content' as the key to match Langchain's default.
        point = models.PointStruct(
            id=doc["id"],
            vector=normalized_embedding.tolist(),
            payload={"page_content": doc["content"]}
        )
        points_to_upsert.append(point)

client.upsert(
    collection_name=collection_name,
    points=points_to_upsert,
    wait=True
)
print(f"✅ Successfully upserted {len(documents)} documents into '{collection_name}'.")
```

### Searching the Collection (Retrieval)

Now that we have a collection, we can search it.
Here's an example implementation:

```python
import torch
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient

# --- 1. Load the same model and tokenizer ---
model_path = "ibm-granite/granite-embedding-30m-english"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model.eval()
print("Embedding model loaded.")

# --- 2. Connect to the existing Qdrant database ---
client = QdrantClient(path="./qdrant_db")
collection_name = "my_enterprise_collection"
print("Connected to Qdrant.")

# --- 3. Define our search query ---
query_text = "How can I run a large model on my computer?"

# --- 4. Embed the query ---
with torch.no_grad():
    tokenized_query = tokenizer(query_text, padding=True, truncation=True, return_tensors='pt')
    model_output = model(**tokenized_query)
    query_embedding = model_output.last_hidden_state[:, 0]
    normalized_query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)

# --- 5. Perform the search ---
search_results = client.search(
    collection_name=collection_name,
    query_vector=normalized_query_embedding.tolist(), # Use the first (and only) vector
    limit=3  # Return the top 3 most similar results
)

# --- 6. Print the results ---
print(f"\nSearch results for: '{query_text}'")
for i, result in enumerate(search_results):
    print(f"Result {i+1}:")
    print(f"  - ID: {result.id}")
    print(f"    Score: {result.score:.4f}")  # The similarity score
    print(f"    Text: {result.payload['text']}")
    print("-" * 30)
```

Running this will likely retrieve the document about Ollama, as it is the most semantically related document to the query in our small database. This process of turning a query into a vector and finding similar documents is the "Retrieval" in Retrieval-Augmented Generation (RAG).

### Conclusion

Now that we understand the basics, it's time to move onto Langchain to integrate both an LLM and a Vector Store for RAG.

## Additional Notes

Here's the full code start to finish:

```python
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

# Create the collection, overwriting if it already exists
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE
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
```
