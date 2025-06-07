---
layout: default
title: "Part 4: RAG with Langchain"
---

## Table of Contents

- [What is Langchain?](#what-is-langchain)
- [Setting up the Langchain Environment](#setting-up-the-langchain-environment)
- [Building a RAG Pipeline](#building-a-rag-pipeline)
- [Full RAG Script](#full-rag-script)
- [How it Works](#how-it-works)

### What is Langchain?

Langchain is a framework designed to simplify the development of applications powered by large language models. It provides the "glue" that connects our different components—the LLM, the vector database, and our application logic—into a coherent pipeline.

While we could write the logic to query Qdrant and then manually pass the results to Ollama ourselves, Langchain provides standardized, high-level interfaces that make this process much easier and more robust.

### Setting up the Langchain Environment

First, we need to install all the necessary Langchain packages to communicate with our components.

```bash
# Core langchain library
pip install langchain

# Community packages for our specific integrations
pip install langchain-community

# Package for interacting with Ollama
pip install langchain-ollama

# Package for interacting with Qdrant
pip install langchain-qdrant

# Package for interacting with HuggingFace
pip install langchain-huggingface
```

Or you can do them all at once with:

```bash
pip install langchain langchain-community langchain-ollama langchain-qdrant langchain-huggingface
```

You should have transformers, torch, and qdrant-client already installed from the previous section.

### Building a RAG Pipeline

A RAG pipeline in Langchain consists of several key components chained together. We will define each one and then combine them using the Langchain Expression Language (LCEL).

1. **The Embedding Model**

As established in Part 3, we will use the HuggingFaceEmbeddings class to load our IBM Granite model. This will be used by the retriever to embed the user's query.

2. **The Vector Store and Retriever**

Langchain has a Qdrant class that can directly connect to our existing vector store. We initialize it by providing the client, the collection name, and our embedding model instance. We can then expose the vector store as a retriever, which is a simple component that "retrieves" documents based on a query.

3. **The LLM**

We use the ChatOllama class to connect to our locally running language model (e.g., Llama 3.2).

4. **The Prompt Template**

This is a crucial step. We create a prompt template that instructs the LLM on how to behave. It must include placeholders for the context (the documents retrieved from Qdrant) and the question (the user's original query). This guides the model to answer based only on the provided documents.

5. **The Output Parser**

Finally, we use a simple StrOutputParser to ensure the final output from the LLM is a clean string.

### Full RAG Script

This script combines all the concepts above into a single, runnable file. It assumes you have already populated your Qdrant database using the script from Part 3.

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient

# --- 1. Initialize Components ---

# Embedding Model
print("Loading embedding model...")
embed_model = HuggingFaceEmbeddings(
    model_name="ibm-granite/granite-embedding-30m-english"
)

# Qdrant Vector Store
client = QdrantClient(path="./qdrant_db")
collection_name = "example_collection"

# Connect Langchain to our Qdrant collection
vector_store = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embed_model,
)

# The Retriever
retriever = vector_store.as_retriever()
print("Retriever initialized.")

# The LLM
llm = ChatOllama(model="llama3") # Use your actual model name
print("Ollama LLM initialized.")

# The Prompt Template
template = """
Answer the question based only on the following context.
If you don't know the answer, just say that you don't know.

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)


# --- 2. Build the RAG Chain using LCEL ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n--- RAG Chain is Ready ---")

# --- 3. Run the Chain ---
question = "What is Qdrant?"
print(f"Asking: {question}")

response = rag_chain.invoke(question)

print("\n--- Answer ---")
print(response)
```

### How it Works

When you call rag_chain.invoke(question), the following happens automatically:

1. The question is passed to the retriever. Simultaneously, RunnablePassthrough() just passes the question through to the next step.
2. The retriever embeds the question using the Granite model and queries the Qdrant database, fetching the most relevant documents.
3. The retrieved documents (as context) and the original question are fed into the prompt template.
4. The fully formatted prompt is sent to the llm (Ollama).
5. The LLM generates an answer based only on the context provided.
6. The StrOutputParser cleans up the LLM's output into a simple string.
