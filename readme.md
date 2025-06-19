# pyscrai & concordia

This repository contains the **pyscrai** package and the core **concordia** framework it aims to leverage. 

The project focuses on building modular simulations that can be easily extended.

## Setup
1. Create a Python 3.12 virtual environment named `.venv` at the project root:
   ```bash
   python -m venv .venv
   ```
2. Activate the virtual environment:
   - On Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows (bash):
     ```bash
     source .venv/Scripts/activate
     ```
3. Install python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   **Optional: For local embedding models**
   - For CUDA/GPU support (local models with GPU acceleration):
     ```bash
     pip install -r requirements-cuda.txt
     ```
   - For CPU-only PyTorch (local models without GPU):
     ```bash
     pip install -r requirements-cpu.txt
     ```
   
   > **Note**: The base installation uses Hugging Face's free Inference API for embeddings. Local model support is optional for offline use or when you prefer full control over the embedding pipeline.
4. Install modules using setup.py:
   ```bash
   pip install -e . 
   ```

5. **Environment Configuration**: Create a `.env` file in the project root with your API keys:
   ```bash
   # Required for Hugging Face embeddings (free)
   HF_API_TOKEN=your_huggingface_token_here
   
   # Required for OpenRouter LLM access
   OPENROUTER_API_KEY=your_openrouter_key_here
   
   # Optional: For local model downloads (if using local embeddings)
   HF_HOME=./models
   ```

## Embedding Options

This project supports two embedding approaches:

### Cloud Embeddings (Default - Recommended)
- **Provider**: Hugging Face Inference API (free)
- **Models**: BGE, E5, all-MiniLM, all-mpnet
- **Requirements**: Base installation only
- **Setup**: Add `HF_API_TOKEN` to `.env` file

### Local Embeddings (Optional)
- **Provider**: SentenceTransformers (offline)
- **Models**: Any compatible model from Hugging Face
- **Requirements**: Install `requirements-cpu.txt` or `requirements-cuda.txt`
- **Use cases**: Privacy, offline operation, custom models

## AI/ML Capabilities

This project provides flexible AI/ML integration with support for both cloud and local inference:

### Language Models
- **Provider**: OpenRouter API (supports 100+ models)
- **Integration**: via `concordia/language_model/openrouter_model.py`
- **Free Options**: Multiple free models available (Mistral, Meta-Llama, etc.)

### Embedding Models
- **Primary**: Hugging Face Inference API (free, no local compute)
  - BGE models (multilingual, high performance)
  - E5 models (general purpose)
  - MiniLM models (lightweight, fast)
- **Fallback**: Local SentenceTransformers (privacy, offline)
  - Requires optional PyTorch installation
  - Full model control and customization

### Vector Database
- **ChromaDB**: Built-in persistence, simple API
- **Use Cases**: Document search, semantic memory, RAG systems

This architecture enables:
- **Rapid prototyping** with cloud APIs (base install)
- **Production deployment** with local models (optional requirements)
- **Hybrid approaches** mixing cloud LLMs with local embeddings

## Documentation

### Core Framework Documentation
- **[Concordia Overview](docs/concordia/concordia_overview.md)** - High-level architecture and component overview
- **[Concordia Developer's Guide](docs/concordia_developers_guide.md)** - Comprehensive development documentation
- **[Integration Test Guide](concordia/concordia_integration_test.md)** - Test validation and examples

### Development Documentation
- **[Development Plan](docs/dev_plan.md)** - Current status and development roadmap
- **[Project Instructions](AGENTS.md)** - Project setup and development guidelines

### API Documentation
- **[API Reference](concordia/.api_docs/README.md)** - Complete auto-generated API documentation with LLM-compatible JSON format
- Source code docstrings provide detailed API documentation for all modules



