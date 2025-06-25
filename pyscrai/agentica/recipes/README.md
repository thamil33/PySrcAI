# PySCRAI Recipe Examples

This directory contains practical examples and setup scripts for pyscrai.agentica.

## Available Recipes

### 1. Default RAG Setup (`default_rag_setup.py`)
Complete example demonstrating how to set up a RAG system for LangChain documentation.

**Features:**
- Automatic environment validation
- Sample document creation if real docs aren't available
- Interactive testing with multiple queries
- Demonstration of ingestion, query, and retrieval

**Usage:**
```bash
# Set environment variables first
export OPENROUTER_API_KEY="your_key_here"

# Run the example
python recipes/default_rag_setup.py
```

### 2. Download LangChain Docs (`download_langchain_docs.py`)
Script to download LangChain documentation from GitHub repositories.

**Features:**
- Downloads official LangChain documentation
- Creates comprehensive sample docs as fallback
- Organizes docs for optimal RAG ingestion
- Handles network failures gracefully

**Usage:**
```bash
# Download to default location (docs/langchain)
python recipes/download_langchain_docs.py

# Download to custom location
python recipes/download_langchain_docs.py --output-dir /path/to/docs
```

### 3. Configuration Examples (`config_examples.py`)
Collection of YAML configuration templates for different scenarios.

**Includes:**
- **LangChain Docs Config**: Optimized for LangChain documentation
- **Local Model Config**: Using offline models (LMStudio + Sentence Transformers)
- **Production Config**: Advanced settings for production deployment
- **Research Config**: Specialized for academic/technical content

**Usage:**
```bash
# Generate example config files
python recipes/config_examples.py

# Use a generated config
python -m pyscrai.agentica.cli --config pyscrai/recipes/configs/langchain_docs.yml --interactive
```

### 4. OpenRouter LangGraph Example (`openrouter_langgraph_example.py`)
Demonstrates integration with LangGraph for complex workflows.

## Quick Start Guide

1. **Environment Setup:**
   ```bash
   # Required environment variables
   export OPENROUTER_API_KEY="your_openrouter_api_key"
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Download Documentation:**
   ```bash
   python recipes/download_langchain_docs.py
   ```

3. **Generate Config Examples:**
   ```bash
   python recipes/config_examples.py
   ```

4. **Run Default RAG Setup:**
   ```bash
   python recipes/default_rag_setup.py
   ```

5. **Try Interactive Mode:**
   ```bash
   python -m pyscrai.agentica.cli --config recipes/configs/langchain_docs.yml --interactive
   ```

## Configuration Templates

The `configs/` subdirectory contains ready-to-use YAML configurations:

- `langchain_docs.yml`: For LangChain documentation RAG
- `local_model.yml`: For completely offline setups
- `production.yml`: For production deployments
- `research.yml`: For academic/research use cases

## Customization

All recipes are designed to be easily customizable:

1. **Modify configurations** in the YAML files
2. **Adjust document paths** in the scripts
3. **Change model settings** for your specific needs
4. **Add custom preprocessing** hooks as needed

## Troubleshooting

**Common Issues:**

1. **Missing API Keys**: Ensure environment variables are set
2. **Network Issues**: Use sample docs if downloads fail
3. **Model Loading**: Check model names and availability
4. **Vector Store**: Ensure sufficient disk space for embeddings

**Getting Help:**

- Check the main README.md for general setup
- Review error messages for specific guidance
- Use `--verbose` flag for detailed logging
