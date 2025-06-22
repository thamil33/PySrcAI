# PyScRAI - Enhanced RAG Development Framework

PyScRAI is a comprehensive Retrieval-Augmented Generation (RAG) framework that provides easy-to-use tools for building, configuring, and deploying RAG applications.

## Features

- ✅ **Vector Store Integration**: ChromaDB support with persistent storage
- ✅ **Document Ingestion**: Automatic loading and chunking for text, markdown, and JSON files
- ✅ **Multiple LLM Providers**: Support for OpenRouter and LMStudio
- ✅ **Flexible Embeddings**: local sentence-transformers
- ✅ **Configuration Management**: YAML-based configuration with templates
- ✅ **CLI Interface**: Command-line tools for ingestion, querying, and interactive mode
- ✅ **Extensible Architecture**: Easy to extend with new adapters and components

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Basic Usage

#### Using the CLI

```bash
# Show vector store information
python -m cli --template default --info

# Ingest documents
python -m cli --template default --ingest ./docs

# Ask a question
python -m cli --template default --query "What is RAG?"

# Start interactive mode
python -m cli --template default --interactive
```

#### Using the Python API

```python
from pyscrai.agents.builder import AgentBuilder
from pyscrai.config.config import AgentConfig

# Create agent with default configuration
agent = AgentBuilder.create_default()

# Or use a template
agent = AgentBuilder.from_template("local_models")

# Ingest documents
doc_ids = agent.ingest(["./docs", "./examples"])

# Query the agent
response = agent.query("How do I create a RAG agent?")
print(response)

# Get response with sources
result = agent.query_with_sources("What are the main components?")
print(f"Answer: {result['answer']}")
for doc in result['source_documents'][:3]:
    print(f"Source: {doc.metadata['source']}")
```

### 3. Configuration

Create a custom configuration file:

```yaml
# config.yaml
models:
  provider: "openrouter"
  model: "mistralai/mistral-small-3.1-24b-instruct:free"
  model_kwargs:
    temperature: 0.7
    max_tokens: 500

vectordb:
  persist_directory: "./my_vector_storage"
  collection_name: "my_docs"

chunking:
  chunk_size: 512
  overlap: 50

rag:
  top_k: 5
  similarity_threshold: 0.7

system_prompt: |
  You are a helpful assistant for my specific domain.
  Answer questions based on the provided context.

data_paths:
  - "./my_docs"
  - "./additional_sources"
```

Use with CLI:

```bash
python -m cli --config config.yaml --interactive
```

### 4. Available Templates

- `default`: OpenRouter 
- `local_models`: LMStudio + local sentence-transformers (fully local)
- `openrouter`: Optimized for OpenRouter API usage

List available templates:

```bash
python -m cli --template default --info
```

## Environment Variables

Set these environment variables for API access:

```bash
export OPENROUTER_API_KEY="your_openrouter_key"
```

## Project Structure

```
pyscrai/
├── adapters/           # LLM, embedding, and vector store adapters
│   ├── llm/           # OpenRouter, LMStudio adapters
│   ├── embeddings/    # sentence-transformers
│   └── vectorstore/   # ChromaDB adapter
├── agents/            # RAG agent implementations
├── cli/               # Command-line interface
├── config/            # Configuration management
│   └── templates/     # Pre-built configuration templates
├── ingestion/         # Document loading and processing
└── tests/             # Unit tests
```

## Advanced Usage

### Custom Preprocessing

```python
from pyscrai.agents.builder import AgentBuilder

def my_preprocessing_hook(document):
    # Custom document preprocessing
    document.metadata['processed'] = True
    return document

agent = AgentBuilder.create_default()
agent.add_preprocessing_hook(my_preprocessing_hook)
```

### Direct Vector Store Operations

```python
# Search without generating answers
results = agent.search_documents("query", k=10, with_scores=True)

# Clear vector store
agent.clear_store()

# Get vector store statistics
info = agent.get_store_info()
print(f"Documents: {info['count']}")
```

## Development

### Running Tests

```bash
pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Roadmap

- [ ] Additional vector store backends (Pinecone, Weaviate)
- [ ] More LLM providers (Anthropic, Azure OpenAI)
- [ ] Advanced retrieval strategies
- [ ] Web UI dashboard
- [ ] Integration with LangMem and Concordia frameworks
