# RAG Agent Directory Structure

The RAG Agent framework has been reorganized into a clean, modular structure:

```
pyscrai/rag_agent/
├── __init__.py                 # Main module interface
├── __main__.py                 # CLI entry point
├── config.yaml                 # Default configuration
├── README.md                   # Quick reference
│
├── src/                        # Core source code
│   ├── base_rag_agent.py       # Abstract base class
│   ├── rag_agent_builder.py    # Builder pattern & factories
│   ├── config_loader.py        # Configuration management
│   ├── rag_pipeline.py         # Legacy pipeline (compatibility)
│   ├── chunking.py             # Document chunking strategies
│   └── cli.py                  # Command-line interface
│
├── adapters/                   # Component adapters
│   ├── __init__.py
│   ├── embedding_adapter.py    # HF API + local embeddings
│   ├── vector_db_adapter.py    # ChromaDB integration
│   └── llm_adapter.py          # OpenRouter via Concordia
│
├── agents/                     # Pre-built specialized agents
│   ├── __init__.py
│   ├── concordia_assistant.py  # Concordia framework expert
│   └── openrouter_assistant.py # OpenRouter API expert
│
├── templates/                  # Examples and templates
│   ├── custom_agent_example.py # Usage examples
│   └── custom_config_template.yaml # Config template
│
└── docs/                       # Documentation
    ├── assistant_development.md # Original development plan
    └── MIGRATION.md            # Migration summary
```

## Benefits of This Structure

### 🗂️ **Organized**
- **`src/`** - Core logic and implementations
- **`adapters/`** - Pluggable components 
- **`agents/`** - Domain-specific implementations
- **`templates/`** - Examples and quickstart
- **`docs/`** - Documentation and guides

### 🔧 **Maintainable**
- Clear separation of concerns
- Easy to locate specific functionality
- Logical grouping of related components

### 🚀 **Developer-Friendly**
- Templates for quick development
- Examples for common patterns
- Documentation co-located with code

### 📦 **Extensible**
- New adapters go in `adapters/`
- New agent types go in `agents/`
- New examples go in `templates/`

## Import Patterns

### Main Interface (Recommended)
```python
# Use the main module interface
from pyscrai.rag_agent import RAGAgentBuilder, create_agent, quick_agent

# Pre-built agents
from pyscrai.rag_agent import ConcordiaAssistant, OpenRouterAssistant
```

### Direct Access (Advanced)
```python
# Direct access to core components
from pyscrai.rag_agent.src.base_rag_agent import BaseRAGAgent
from pyscrai.rag_agent.adapters.embedding_adapter import EmbeddingAdapter
```

### CLI Usage
```bash
# Module entry point
python -m pyscrai.rag_agent --agent-type concordia --interactive
```

This structure supports both simple usage (through the main interface) and advanced customization (through direct component access) while keeping the codebase well-organized and maintainable.
