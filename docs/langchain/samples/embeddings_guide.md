
# Embeddings in LangChain

## What are Embeddings?
Embeddings are numerical representations of text that capture semantic meaning.

## Embedding Providers

### OpenAI Embeddings
```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key="your-api-key"
)
```

### HuggingFace Embeddings
```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### HuggingFace Hub
```python
from langchain.embeddings import HuggingFaceHubEmbeddings

embeddings = HuggingFaceHubEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token="your-token"
)
```

## Usage Examples

### Embedding Documents
```python
# Embed multiple documents
doc_embeddings = embeddings.embed_documents([
    "This is a document",
    "This is another document"
])
```

### Embedding Queries
```python
# Embed a query
query_embedding = embeddings.embed_query("What is machine learning?")
```

## Best Practices

1. **Model Selection**: Choose models appropriate for your domain
2. **Consistency**: Use the same embedding model for indexing and querying
3. **Normalization**: Some models benefit from L2 normalization
4. **Batch Processing**: Process multiple texts together for efficiency
5. **Caching**: Cache embeddings to avoid recomputation
