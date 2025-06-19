# RAG Agent Framework

A flexible, extensible RAG (Retrieval-Augmented Generation) agent system with builder pattern support for creating specialized assistants with custom configurations, instructions, and data sources.

## Features

- 🏗️ **Builder Pattern**: Fluent interface for creating custom agents
- 🔌 **Multiple Providers**: Support for cloud (HuggingFace API) and local embeddings
- 📚 **Flexible Ingestion**: Support for various document formats and chunking strategies
- 🎯 **Specialized Agents**: Pre-built agents for specific domains (Concordia, OpenRouter)
- ⚙️ **Configurable**: YAML-based configuration with sensible defaults
- 🚀 **Easy Integration**: Simple CLI and programmatic interfaces

## Quick Start

### Using Pre-built Agents
```python
from pyscrai.rag_agent import create_agent

# Create a Concordia expert
agent = create_agent("concordia")
agent.ingest_documents(["docs/concordia/"])
response = agent.query("How do I create a basic entity?")
```

### Building Custom Agents
```python
from pyscrai.rag_agent import RAGAgentBuilder

agent = (RAGAgentBuilder()
         .with_name("MyDocAgent")
         .with_system_prompt("You are a helpful assistant...")
         .with_data_sources(["docs/", "README.md"])
         .build())
```

### Quick Agent Creation
```python
from pyscrai.rag_agent import quick_agent

agent = quick_agent(
    name="CodeReviewer", 
    system_prompt="You are an expert code reviewer...",
    data_sources=["src/", "docs/standards.md"]
)
```

## CLI Usage

```bash
# Interactive mode with Concordia assistant
python -m pyscrai.rag_agent --agent-type concordia --interactive

# Create custom agent
python -m pyscrai.rag_agent --agent-type custom \
    --name "MyAgent" \
    --system-prompt "You are helpful..." \
    --data-sources docs/ --interactive
```

## Installation Options

**Base (Cloud embeddings - free)**
```bash
pip install -r requirements.txt
```

**With local embeddings**
```bash
pip install -r requirements-cpu.txt  # CPU only
pip install -r requirements-cuda.txt # GPU support
```

See `templates/` for examples and configuration templates.
