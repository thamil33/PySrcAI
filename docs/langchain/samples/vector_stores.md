
# Vector Stores in LangChain

## Overview
Vector stores enable efficient storage and retrieval of document embeddings.

## Supported Vector Stores

### Chroma
Local, persistent vector database:

```python
from langchain.vectorstores import Chroma

vectorstore = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
```

### Pinecone
Cloud-based vector database:

```python
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(api_key="your-api-key", environment="us-west1-gcp")
vectorstore = Pinecone.from_documents(documents, embeddings, index_name="my-index")
```

### FAISS
Facebook AI Similarity Search:

```python
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("./faiss_index")
```

## Vector Store Operations

### Adding Documents
```python
# Add new documents
vectorstore.add_documents(new_documents)

# Add texts directly
vectorstore.add_texts(texts, metadatas=metadatas)
```

### Searching
```python
# Similarity search
docs = vectorstore.similarity_search("query", k=5)

# Similarity search with scores
docs_with_scores = vectorstore.similarity_search_with_score("query", k=5)

# Maximum marginal relevance search
docs = vectorstore.max_marginal_relevance_search("query", k=5)
```

### Filtering
```python
# Search with metadata filters
docs = vectorstore.similarity_search(
    "query",
    filter={"source": "document.pdf"}
)
```
