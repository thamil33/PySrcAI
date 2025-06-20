# pyscrai RAG Agent System

## Overview

pyscrai provides a flexible, config-driven Retrieval-Augmented Generation (RAG) agent system for Python 3.12. The system is designed for research, prototyping, and production use, with a focus on:

- **Interactive CLI** for conversational and retrieval-based tasks
- **Centralized YAML config** as the single source of truth for agent setup
- **Pluggable embedding and LLM backends** (HuggingFace, SentenceTransformers, OpenRouter)
- **Easy extensibility and debugging**

---

## Key Features

- **Interactive CLI**: Launch an agent and interact with it in a conversational shell. Supports ingestion, querying, and database management.
- **Config-Driven**: All agent setup (system prompt, data sources, model/embedding/vector DB settings) is defined in a YAML config file. No hardcoded agent types.
- **Embeddings**: Use either HuggingFace API or local SentenceTransformers (with CPU or CUDA) for document and query embeddings.
- **LLM Integration**: Uses the OpenRouter API for language models, via the `language_model_client` in `pyscrai/language_model_client`.
- **Python 3.12**: Fully compatible and tested with Python 3.12.

---

## Installation

1. **Clone the repository** and enter the project directory.

2. **Install dependencies**:
     - For API-only (no local models):
     ```sh
     pip install -r requirements.txt
     ```

3. **Install the package in editable mode**:
   ```sh
   pip install -e .
   ```

---

## Usage: Interactive CLI

1. **Prepare your config** (see `pyscrai/rag_agent/templates/concordia.yaml` for an example). This config defines your agent's system prompt, data sources, embedding/LLM/vector DB settings, and more.

2. **Launch the CLI**:
   ```sh
   python -m pyscrai.rag_agent --config pyscrai/rag_agent/templates/openrouter.yaml --interactive
   ```
   - You can also use your own config file.

3. **CLI Features**:
   - Ingest documents (auto or manual)
   - Query the agent
   - View collection info
   - Clear/reset the vector database
   - All config accesses are logged for debugging (enable with `--log-config-access`)

---

## Embedding Backends
- **HuggingFace API**: Set `provider: huggingface_api` in your config. Requires a HuggingFace API key.
- **SentenceTransformers (local)**: Set `provider: local_sentencetransformers` and choose a model. Works with CPU or CUDA (set `device: cpu` or `device: cuda`).

## LLM Backend
- **OpenRouter API**: All LLM calls are routed through the OpenRouter API using the `language_model_client` in `pyscrai/language_model_client`. Set your API key and model in the config.

---

## Example Config (YAML)
```yaml
models:
  language_model: "google/gemini-2.5-flash-lite-preview-06-17"
embedding:
  provider: "local_sentencetransformers"
  model: "sentence-transformers/all-mpnet-base-v2"
  device: "cuda"
vector_db:
  type: "chromadb"
  persist_directory: "./vector_storage"
chunking:
  chunk_size: 512
  overlap: 50
rag:
  top_k: 5
  similarity_threshold: 0.7
openrouter:
  api_key_env: "OPENROUTER_API_KEY"
  base_url: "https://openrouter.ai/api/v1"
  model_kwargs:
    temperature: 1
    max_tokens: 100
agent:
  name: "OpenRouterAssistant"
  system_prompt: "You are a Openrouter API expert..."
  data_sources:
    - path: "docs/references/openrouter_docs.txt"
      type: "text"
```

---

## Notes
- All agent logic is config-driven—no agent_type or hardcoded classes.
- For best results, set up your API keys as environment variables.
- See the `pyscrai/rag_agent/templates/` directory for more config examples.

---

For more details, see the code and templates in the repository.
