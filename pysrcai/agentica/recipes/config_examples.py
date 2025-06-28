"""
RAG Configuration Examples for pysrcai

This file contains various configuration examples for different RAG scenarios.
Copy and modify these configs for your specific use cases.
"""

# Example 1: Default LangChain Documentation RAG
LANGCHAIN_DOCS_CONFIG = """
# LangChain Documentation RAG Configuration
models:
  provider: "openrouter"
  model: "mistralai/mistral-small-3.1-24b-instruct:free"
  temperature: 0.1
  max_tokens: 2000

embedding:
  provider: "local_sentencetransformers"
  model: "all-MiniLM-L6-v2"

vectordb:
  provider: "chroma"
  persist_directory: "./vectorstore/langchain_docs"
  collection_name: "langchain_documentation"
  anonymized_telemetry: false

chunking:
  strategy: "recursive"
  chunk_size: 1000
  overlap: 200

rag:
  top_k: 5
  search_type: "similarity"
  system_prompt: |
    You are a helpful assistant that answers questions about LangChain.
    Use the provided context to answer questions accurately and cite sources when possible.
    If you don't know something, say so rather than making up information.

agent:
  name: "LangChain Documentation Assistant"
  description: "RAG assistant for LangChain framework documentation"
  data_paths:
    - "./docs/langchain"
"""

# Example 2: Local Model with Sentence Transformers
LOCAL_MODEL_CONFIG = """
# Local Model RAG Configuration (No API keys required)
models:
  provider: "lmstudio"
  model: "local-model"
  base_url: "http://localhost:1234/v1"
  temperature: 0.3
  max_tokens: 1500

embedding:
  provider: "local_sentencetransformers"
  model: "all-MiniLM-L6-v2"
  device: "cpu"

vectordb:
  provider: "chroma"
  persist_directory: "./vectorstore/local_docs"
  collection_name: "local_documentation"

chunking:
  strategy: "recursive"
  chunk_size: 800
  overlap: 150

rag:
  top_k: 3
  search_type: "similarity"
  system_prompt: |
    You are a knowledgeable assistant. Answer questions based on the provided context.
    Be concise and accurate in your responses.

agent:
  name: "Local RAG Assistant"
  description: "Local RAG assistant using offline models"
  data_paths:
    - "./docs"
"""

# Example 3: Production RAG with Advanced Settings
PRODUCTION_CONFIG = """
# Production RAG Configuration
models:
  provider: "openrouter"
  model: "openai/gpt-4-turbo"
  temperature: 0.2
  max_tokens: 3000
  model_kwargs:
    frequency_penalty: 0.1
    presence_penalty: 0.1

embedding:
  provider: "local_sentencetransformers"
  model: "all-mpnet-base-v2"
  fallback_models:
    - "all-MiniLM-L6-v2"
    - "all-mpnet-base-v2"

vectordb:
  provider: "chroma"
  persist_directory: "./vectorstore/production"
  collection_name: "knowledge_base"
  anonymized_telemetry: false
  settings:
    hnsw_space: "cosine"

chunking:
  strategy: "semantic"
  chunk_size: 1200
  overlap: 300
  separators:
    - "\\n\\n"
    - "\\n"
    - "\\. "
    - " "

rag:
  top_k: 7
  search_type: "mmr"  # Maximum Marginal Relevance
  mmr_diversity_score: 0.3
  system_prompt: |
    You are an expert AI assistant with access to a comprehensive knowledge base.

    Instructions:
    1. Answer questions accurately using the provided context
    2. If multiple sources are relevant, synthesize information appropriately
    3. Always cite sources using [Source: filename] format
    4. If information is insufficient, acknowledge limitations
    5. Provide structured, well-formatted responses
    6. Include relevant examples when helpful

    Context: {context}

    Question: {question}

    Answer:

agent:
  name: "Production RAG Assistant"
  description: "High-performance RAG assistant for production use"
  data_paths:
    - "./data/documents"
    - "./data/manuals"
    - "./data/guides"
  openrouter_api_key_env: "OPENROUTER_API_KEY"
"""

# Example 4: Research Assistant Configuration
RESEARCH_CONFIG = """
# Research Assistant RAG Configuration
models:
  provider: "openrouter"
  model: "anthropic/claude-3-haiku"
  temperature: 0.1
  max_tokens: 4000

embedding:
  provider: "local_sentencetransformers"
  model: "e5-large-v2"

vectordb:
  provider: "chroma"
  persist_directory: "./vectorstore/research"
  collection_name: "research_papers"

chunking:
  strategy: "recursive"
  chunk_size: 1500
  overlap: 400  # Higher overlap for academic content

rag:
  top_k: 10  # More context for research
  search_type: "similarity"
  system_prompt: |
    You are a research assistant specializing in academic and technical content.

    Guidelines:
    1. Provide comprehensive, well-researched answers
    2. Include specific citations and references
    3. Explain complex concepts clearly
    4. Highlight key findings and methodologies
    5. Note any limitations or gaps in the available information
    6. Suggest related topics for further exploration

    Always maintain academic rigor and objectivity in your responses.

agent:
  name: "Research Assistant"
  description: "AI assistant for academic and technical research"
  data_paths:
    - "./research/papers"
    - "./research/reports"
"""

def save_config_examples():
    """Save all configuration examples to files."""
    from pathlib import Path

    configs_dir = Path("recipes/configs")
    configs_dir.mkdir(exist_ok=True)

    configs = {
        "langchain_docs.yml": LANGCHAIN_DOCS_CONFIG,
        "local_model.yml": LOCAL_MODEL_CONFIG,
        "production.yml": PRODUCTION_CONFIG,
        "research.yml": RESEARCH_CONFIG
    }
    for filename, content in configs.items():
        config_path = configs_dir / filename
        config_path.write_text(content.strip())
        print(f"Created: {config_path}")

    print(f"\nConfiguration examples saved to: {configs_dir}")
    print("Copy and modify these configs for your specific use cases.")


if __name__ == "__main__":
    save_config_examples()
