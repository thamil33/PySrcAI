# Concordia Assistant

A RAG (Retrieval-Augmented Generation) powered AI assistant built using Concordia framework components. The assistant provides intelligent responses about Concordia development, OpenRouter API usage, and related documentation.

## Features

- **Dual Embedding Support**: Choose between local SentenceTransformers or HuggingFace API embeddings
- **ChromaDB Vector Storage**: Persistent vector database for efficient document retrieval
- **Intelligent Chunking**: Hierarchical chunking for JSON files, semantic chunking for text
- **OpenRouter Integration**: Uses Concordia's OpenRouter client for free model access
- **Interactive CLI**: Terminal-based chat interface with query and status commands
- **Configurable**: YAML-based configuration for easy customization

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file or set these environment variables:

```bash
OPENROUTER_API_KEY=your_openrouter_key_here
HF_API_TOKEN=your_huggingface_token_here  # Optional, for HuggingFace embeddings
HF_HOME=C:\Users\your_username\.cache\huggingface\hub  # For local models
```

### 3. Initialize the Assistant

```bash
python -m pyscrai.concordia_assistant init
```

This will:
- Download and cache the SentenceTransformers model (if using local embeddings)
- Create the ChromaDB vector database
- Ingest and chunk all configured documentation

### 4. Start Chatting

```bash
python -m pyscrai.concordia_assistant chat
```

## Configuration

The assistant is configured via `config.yaml`. Key settings include:

### Embedding Provider
Choose between local SentenceTransformers (recommended) or HuggingFace API:

```yaml
embedding:
  provider: "sentence_transformers"  # or "huggingface"
```

### Data Sources
Configure which documentation to ingest:

```yaml
data_sources:
  - path: "docs/references/openrouter_docs.txt"
    type: "text"
  - path: "concordia/.api_docs/_build/json/"
    type: "json_directory"
```

## Usage Examples

### Command Line

```bash
# Ask a single question
python -m pyscrai.concordia_assistant query "How do I create a basic entity in Concordia?"

# Check status
python -m pyscrai.concordia_assistant status

# Re-ingest documents
python -m pyscrai.concordia_assistant init --force
```

### Interactive Chat

```bash
python -m pyscrai.concordia_assistant chat
```

Available chat commands:
- `exit`, `quit`, `q`: Exit the chat
- `help`: Show help message
- `status`: Show assistant status
- `clear`: Clear the screen

### Programmatic Usage

```python
from pyscrai.concordia_assistant import ConcordiaRAGPipeline

# Initialize the pipeline
pipeline = ConcordiaRAGPipeline()
pipeline.initialize()

# Ingest documents (if not already done)
pipeline.ingest_documents()

# Ask questions
response = pipeline.query("How do I use OpenRouter with Concordia?")
print(response)
```

## Architecture

The assistant consists of several key components:

1. **Config Loader**: Manages YAML configuration and environment variables
2. **Embedding Adapter**: Handles both SentenceTransformers and HuggingFace embeddings
3. **Vector Database**: ChromaDB adapter for document storage and retrieval
4. **Document Chunker**: Intelligent chunking strategies for different file types
5. **LLM Adapter**: OpenRouter integration using Concordia's implementation
6. **RAG Pipeline**: Orchestrates all components for query processing
7. **CLI Interface**: User-friendly command-line interface

## Performance Notes

- **Local vs. API Embeddings**: SentenceTransformers (local) is typically faster than HuggingFace API
- **GPU Support**: If you have a CUDA-compatible GPU, set `device: "cuda"` in the config
- **Model Caching**: Models are cached locally in `HF_HOME` directory for faster subsequent loads
- **Batch Processing**: Document ingestion processes chunks in batches to manage memory

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`
2. **API Key Issues**: Verify your OpenRouter API key is correctly set in `.env`
3. **Model Download**: First run may take time as models are downloaded and cached
4. **Memory Issues**: For large document sets, consider reducing batch size in the pipeline

### Debug Mode

Enable debug logging by modifying the logging configuration in the relevant modules.

## Extension Points

The assistant is designed to be extensible:

- Add new embedding providers in `embedding_adapter.py`
- Implement different vector databases by creating new adapters
- Extend chunking strategies in `chunking.py`
- Add new data source types in `rag_pipeline.py`

## Future Enhancements

- Query rewriting and expansion
- Re-ranking of retrieved documents
- Conversation memory and context
- Web interface
- Integration with Concordia entity-component system
