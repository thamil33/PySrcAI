# PyScRAI Config Setting Flow

This document details where each setting from a PyScRAI YAML config (e.g., `default.yaml`) is routed in the codebase when processed by `cli.py`.

---

## Config Setting Routing Table

| Config Section / Key         | Consumed By (Class/Function)         | File/Module                        | Usage Context / Purpose                                                                                   |
|------------------------------|--------------------------------------|------------------------------------|----------------------------------------------------------------------------------------------------------|
| `models`                     | `ModelConfig`                        | `config/config.py`                 | Instantiated as part of `AgentConfig`; passed to LLM adapter via `AgentBuilder`                          |
| ├─ `provider`                | `ModelConfig`                        | `config/config.py`                 | Used by `create_llm` to select LLM backend                                                               |
| ├─ `model`                   | `ModelConfig`                        | `config/config.py`                 | Used by LLM adapter to select model                                                                      |
| └─ `model_kwargs`            | `ModelConfig`                        | `config/config.py`                 | Passed as kwargs to LLM adapter                                                                          |
| `embedding`                  | `EmbeddingConfig`                    | `config/config.py`                 | Instantiated as part of `AgentConfig`; passed to embedding adapter via `AgentBuilder`                    |
| ├─ `provider`                | `EmbeddingConfig`                    | `config/config.py`                 | Used by embedding factory to select backend                                                              |
| ├─ `model`                   | `EmbeddingConfig`                    | `config/config.py`                 | Used by embedding adapter to select model                                                                |
| ├─ `device`                  | `EmbeddingConfig`                    | `config/config.py`                 | Used by embedding adapter to select device                                                               |
| ├─ `cache_folder`            | `EmbeddingConfig`                    | `config/config.py`                 | Used by embedding adapter for caching                                                                    |
| └─ `fallback_models`         | `EmbeddingConfig`                    | `config/config.py`                 | Used by embedding adapter for fallback logic                                                             |
| `vectordb`                   | `VectorDBConfig`                     | `config/config.py`                 | Instantiated as part of `AgentConfig`; passed to vector store adapter via `AgentBuilder`                 |
| ├─ `persist_directory`       | `VectorDBConfig`                     | `config/config.py`                 | Used by vector store adapter to set storage location                                                     |
| ├─ `collection_name`         | `VectorDBConfig`                     | `config/config.py`                 | Used by vector store adapter to select collection                                                        |
| ├─ `anonymized_telemetry`    | `VectorDBConfig`                     | `config/config.py`                 | Used by vector store adapter                                                                             |
| └─ `settings`                | `VectorDBConfig`                     | `config/config.py`                 | Passed to vector store adapter (e.g., HNSW params)                                                       |
| `chunking`                   | `ChunkingConfig`                     | `config/config.py`                 | Instantiated as part of `AgentConfig`; used by ingestion pipeline                                        |
| ├─ `json_strategy`           | `ChunkingConfig`                     | `config/config.py`                 | Used by chunker to select JSON chunking strategy                                                         |
| ├─ `text_strategy`           | `ChunkingConfig`                     | `config/config.py`                 | Used by chunker to select text chunking strategy                                                         |
| ├─ `chunk_size`              | `ChunkingConfig`                     | `config/config.py`                 | Used by chunker to set chunk size                                                                        |
| └─ `overlap`                 | `ChunkingConfig`                     | `config/config.py`                 | Used by chunker to set overlap                                                                           |
| `rag`                        | `RAGConfig`                          | `config/config.py`                 | Instantiated as part of `AgentConfig`; used by agent pipeline                                            |
| ├─ `top_k`                   | `RAGConfig`                          | `config/config.py`                 | Used by agent to set number of retrieved docs                                                            |
| ├─ `similarity_threshold`    | `RAGConfig`                          | `config/config.py`                 | Used by agent for filtering                                                                              |
| └─ `enable_reranking`        | `RAGConfig`                          | `config/config.py`                 | Used by agent to enable reranking                                                                        |
| `system_prompt`              | `AgentConfig`                        | `config/config.py`                 | Used by agent as the system prompt for LLM                                                               |
| `data_paths`                 | `AgentConfig`                        | `config/config.py`                 | Used by agent/ingestion pipeline to locate data                                                          |
| `openrouter_api_key_env`     | `AgentConfig`                        | `config/config.py`                 | Used by LLM adapter to fetch API key from environment                                                    |

---

## Assembly & Processing Flow

1. **CLI Argument Parsing**: `cli.py` parses CLI args and loads config (from file or template) via `load_config` or `load_template` in `config/config.py`.
2. **Config Instantiation**: The YAML is parsed into an `AgentConfig` dataclass, which contains all sub-configs.
3. **Agent Construction**: `AgentBuilder` (in `agents/builder.py`) receives the `AgentConfig` and instantiates the agent, passing each config section to the appropriate adapter/factory (LLM, embeddings, vector store, chunker, etc).
4. **Component Initialization**: Each adapter or pipeline component uses its config section to initialize itself and control runtime behavior.

---

## Visual Diagram

```mermaid
graph TD
    A[CLI Args / YAML Config] --> B[load_config / load_template]
    B --> C[AgentConfig]
    C --> D1[ModelConfig] z
    C --> D2[EmbeddingConfig]
    C --> D3[VectorDBConfig]
    C --> D4[ChunkingConfig]
    C --> D5[RAGConfig]
    C --> D6[Other Agent Settings]
    D1 --> E1[LLM Adapter]
    D2 --> E2[Embedding Adapter]
    D3 --> E3[Vector Store Adapter]
    D4 --> E4[Chunker]
    D5 --> E5[RAG Pipeline]
    D6 --> E6[Agent/Runtime]
```

---

*For more details, see the relevant modules in the codebase or ask for a deeper dive on any config key.*
