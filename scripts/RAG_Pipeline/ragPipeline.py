import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

# --- 1. Configuration and Setup ---

print("--- Stage 1: Configuration and Setup ---")

# Define a stable, absolute path for the database
# This creates the 'qdrant_db' folder in the same directory as the script

# Define the model for embeddings and the collection name
embed_model_name = "ibm-granite/granite-embedding-30m-english"
collection_name = "example_collection"
vector_size = 384 # Dimension for the granite-30m-english model

# Initialize Qdrant client
client = QdrantClient(path="./qdrant_db")
print("✅ Qdrant client initialized.")

# Recreate the collection to ensure a fresh start
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

# Load the embedding model
print("Loading embedding model...")
embed_model = HuggingFaceEmbeddings(model_name=embed_model_name)
print("✅ Embedding model loaded.")


# --- 2. Document Population ---

print("\n--- Stage 2: Document Population ---")

# Define the documents to be stored
docs = [
    Document(page_content="The IBM Granite models are a family of open-source language models for enterprise use."),
    Document(page_content="Qdrant is a high-performance vector database and similarity search engine written in Rust."),
    Document(page_content="Ollama allows you to run large language models, like Llama 3, locally on your own machine."),
    Document(page_content="Retrieval-Augmented Generation (RAG) is an architecture that combines a retriever with a generator to answer questions."),
]
print(f"Found {len(docs)} documents to process.")

# 1. Embed the documents' content
texts = [doc.page_content for doc in docs]
embeddings = embed_model.embed_documents(texts)
print(f"✅ Embedded {len(embeddings)} documents.")

# 2. Upsert the embeddings and payloads into Qdrant
client.upsert(
    collection_name=collection_name,
    points=models.Batch(
        ids=list(range(len(texts))), # Assign simple integer IDs
        vectors=embeddings,
        payloads=[{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    ),
    wait=True # Wait for the operation to complete
)
print(f"✅ Successfully upserted {len(docs)} documents into Qdrant.")


# --- 3. Build and Run the RAG Chain ---

print("\n--- Stage 3: Build and Run RAG Chain ---")

# --- KEY CHANGE: Initialize the LangChain wrapper with our EXISTING client ---
vector_store = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embed_model,
)
print("✅ LangChain Qdrant vector store initialized.")

# The Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("Retriever initialized.")

# The LLM
llm = ChatOllama(model="llama3.2", temperature=0)
print("Ollama LLM initialized.")

# The Prompt Template
template = """
Answer the question based only on the following context.
If the context does not contain the answer, state that you don't have enough information.

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Build the RAG Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("✅ RAG Chain is ready.")

# --- 4. Inspect Retriever Output ---

print("\n--- Stage 4: Inspect Retrieved Context ---")

def inspect_retrieval(question_to_ask):
    """
    A helper function to retrieve documents and print them for inspection.
    """
    print(f"Asking: \"{question_to_ask}\"")
    retrieved_docs = retriever.invoke(question_to_ask)
    print("\n--- Context that will be sent to the LLM: ---")
    if not retrieved_docs:
        print("No documents were retrieved.")
    for i, doc in enumerate(retrieved_docs):
        print(f"  Doc {i+1}: {doc.page_content}")
    print("------------------------------------------")
    return retrieved_docs

# Inspect the context for the first question
question_1 = "What is Qdrant?"
inspect_retrieval(question_1)


# --- 5. Run the Full Chain and Get Final Answer ---

print("\n--- Stage 5: Run Full RAG Chain ---")

# Run the full chain for the first question
print(f"Asking (full chain): \"{question_1}\"")
response_1 = rag_chain.invoke(question_1)
print("\n--- Final Answer from LLM: ---")
print(response_1)
print("--------------------------------")


# --- Let's do it again for the second question ---

print("\n\n--- Inspecting and Running Second Question ---")

# Inspect context for the second question
question_2 = "How can I run a large model on my computer?"
inspect_retrieval(question_2)

# Run the full chain for the second question
print(f"\nAsking (full chain): \"{question_2}\"")
response_2 = rag_chain.invoke(question_2)
print("\n--- Final Answer from LLM: ---")
print(response_2)
print("--------------------------------")
