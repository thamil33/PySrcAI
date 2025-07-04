# Memory System Integration Guide

This guide explains how to integrate the embedding-based memory system into PySrcAI simulations.

## Overview

The memory system provides two types of memory banks:
- **BasicMemoryBank**: Simple chronological storage with text-based search
- **AssociativeMemoryBank**: Advanced storage with embedding-based semantic search

## Configuration

### Memory Configuration

```python
from pysrcai.src.config.embedding_config import MemoryConfig, EmbeddingConfig

# Basic memory configuration
basic_config = MemoryConfig(
    type="basic",
    max_memories=1000,
    importance_threshold=0.5,
    max_context_memories=5
)

# Associative memory with embeddings
associative_config = MemoryConfig(
    type="associative",
    max_memories=1000,
    importance_threshold=0.5,
    max_context_memories=5,
    embedding=EmbeddingConfig(
        provider="local_sentencetransformers",
        model="all-MiniLM-L6-v2",
        device="cpu"
    )
)
```

### Embedding Configuration

```python
from pysrcai.src.config.embedding_config import EmbeddingConfig

embedding_config = EmbeddingConfig(
    provider="local_sentencetransformers",
    model="all-MiniLM-L6-v2",  # Fast, lightweight model
    device="cpu",  # or "cuda" for GPU
    cache_folder="./models",  # Optional cache directory
    fallback_models=["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2"]
)
```

## Usage

### Creating Memory Banks

```python
from pysrcai.src.agents.memory.memory_factory import create_memory_bank_with_embeddings

# Create memory bank with automatic embedding setup
memory_bank = create_memory_bank_with_embeddings(config)

# Or create manually
from pysrcai.src.agents.memory.memory_factory import create_memory_bank, create_embedder_from_config

embedder = create_embedder_from_config(embedding_config)
memory_bank = create_memory_bank(config, embedder)
```

### Using Memory Banks

```python
# Add memories
memory_bank.add_memory(
    "Alice greeted Bob warmly",
    tags=['conversation', 'greeting'],
    importance=0.8
)

# Retrieve recent memories
recent = memory_bank.retrieve_recent(5)

# Semantic search (associative memory only)
similar = memory_bank.retrieve_by_query("friendly interaction", k=3)

# Tag-based search
tagged = memory_bank.retrieve_by_tags(['conversation'], k=5)
```

### Using Memory Components

```python
from pysrcai.src.agents.memory.memory_components_v2 import MemoryComponent

# Create memory component
memory_component = MemoryComponent(
    memory_bank=memory_bank,
    memory_importance_threshold=0.5,
    max_context_memories=5
)

# The component automatically:
# - Provides relevant memories as context before actions
# - Stores actions and observations as memories
# - Retrieves related past experiences
```

## Integration with Agents

### Agent Configuration

```yaml
agents:
  - name: Alice
    type: actor
    context_components:
      - name: memory
        type: memory
        memory:
          memory_bank:
            type: associative
            max_memories: 100
            embedding:
              provider: local_sentencetransformers
              model: all-MiniLM-L6-v2
              device: cpu
          importance_threshold: 0.5
          max_context_memories: 5
```

### Programmatic Integration

```python
from pysrcai.src.agents.memory.memory_factory import create_memory_bank_with_embeddings
from pysrcai.src.agents.memory.memory_components_v2 import MemoryComponent

# Create memory bank
memory_config = MemoryConfig(
    type="associative",
    max_memories=100,
    embedding=EmbeddingConfig(
        provider="local_sentencetransformers",
        model="all-MiniLM-L6-v2",
        device="cpu"
    )
)
memory_bank = create_memory_bank_with_embeddings(memory_config)

# Create memory component
memory_component = MemoryComponent(
    memory_bank=memory_bank,
    memory_importance_threshold=0.5,
    max_context_memories=5
)

# Add to agent
agent.add_context_component(memory_component)
```

## Persistence and State Management

### Saving State

```python
# Get component state
state = memory_component.get_state()

# Save to file
import json
with open('memory_state.json', 'w') as f:
    json.dump(state, f)
```

### Loading State

```python
# Load from file
with open('memory_state.json', 'r') as f:
    state = json.load(f)

# Restore component state
memory_component.set_state(state)
```

## Vector Store Integration

For long-term persistence and larger memory capacity, you can integrate with vector stores:

```python
from pysrcai.src.config.embedding_config import VectorDBConfig
from embeddings.vectorstore.factory import create_vectorstore

# Configure vector store
vectorstore_config = VectorDBConfig(
    provider="chroma",
    collection_name="agent_memories",
    persist_directory="./data/vectordb"
)

# Create vector store
embedder = create_embedder_from_config(embedding_config)
vectorstore = create_vectorstore(vectorstore_config, embedder)

# Use for long-term storage
vectorstore.add_texts(
    texts=["Memory text 1", "Memory text 2"],
    metadatas=[{"tags": ["conversation"]}, {"tags": ["action"]}]
)
```

## Performance Considerations

### Model Selection

- **all-MiniLM-L6-v2**: Fast, lightweight (384 dimensions)
- **all-mpnet-base-v2**: Better quality, slower (768 dimensions)
- **all-MiniLM-L12-v2**: Good balance (384 dimensions)

### Memory Management

- Use `max_memories` to limit memory bank size
- Consider using vector stores for long-term persistence
- Implement memory importance scoring for better retention

### Caching

- Set `cache_folder` in embedding config to cache models
- Models are downloaded once and reused

## Troubleshooting

### Common Issues

1. **"Embedder must be set before adding memories"**
   - Ensure embedding config is provided for associative memory
   - Check that sentence-transformers is installed

2. **Import errors**
   - Ensure embeddings directory is in Python path
   - Install required dependencies: `pip install sentence-transformers torch`

3. **Memory leaks**
   - Limit `max_memories` in configuration
   - Use vector stores for large memory banks

### Dependencies

```bash
pip install sentence-transformers torch numpy
pip install chromadb  # For vector store persistence
```

## Testing

Run the integration test:

```bash
python pysrcai/src/agents/memory/test_memory_integration.py
```

This will test both basic and associative memory functionality. 