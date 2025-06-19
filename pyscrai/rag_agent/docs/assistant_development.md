# Concordia Assistant Development Plan

## Project Overview

A RAG-powered AI assistant that uses the Concordia framework to answer questions about OpenRouter API and Concordia development. The agent will ingest documentation and provide intelligent responses using vector search and language model generation.

## Approved Architecture

### Core Technology Stack
- **Vector Database**: ChromaDB (simple API, built-in persistence, good for initial implementation)
- **Embeddings**: 
  - **Primary**: HuggingFace Inference API (free, cloud-based) via `pyscrai/embedding/hf_embedding.py`
  - **Alternative**: Local SentenceTransformers (privacy/offline use)
- **Language Model**: OpenRouter via `concordia/language_model/openrouter_model.py`
- **Target Model**: `mistralai/mistral-small-3.1-24b-instruct:free` (or similar free models)

### Document Processing Strategy
- **JSON Documents**: Hierarchical chunking (leveraging natural structure)
- **Text Files**: Semantic chunking (based on headings/sections)
- **Chunk Configuration**: Configurable via YAML settings

### RAG Implementation
- **Phase 1**: Basic RAG Pipeline (Query → Vector Search → Top k Results → LLM Response)
- **Phase 2**: Advanced features (Query rewriting, reranking, structured formatting)
- **Metadata Enhancement**: Include document types, sections, confidence scores

### Agent Architecture
- **Initial**: Standalone Concordia Language Model consumer (simple approach)
- **Future**: Full Concordia Entity-Component architecture for extensibility

## Implementation Phases

### Phase 1: Document Ingestion & Storage
- [x] Setup ChromaDB integration
- [x] Create document loader for text and JSON files
- [x] Implement HuggingFace embeddings integration
- [x] Add configurable chunking strategy
- [x] YAML configuration system

### Phase 2: Query Processing
- [ ] Integrate OpenRouter model via Concordia client
- [ ] Implement basic RAG retrieval pipeline
- [ ] Create interactive CLI with prompt loop
- [ ] Add response formatting

### Phase 3: Refinements
- [ ] Add query rewriting for better retrieval
- [ ] Implement result reranking
- [ ] Add persistent conversation history
- [ ] Improve response formatting and citations

## Configuration Design

**Configuration File**: `config.yaml`
```yaml
models:
  language_model: "mistralai/mistral-small-3.1-24b-instruct:free"
  embedding_provider: "huggingface_api"  # or "local_sentencetransformers"
  embedding_model: "BAAI/bge-base-en-v1.5"

vector_db:
  type: "chromadb"
  persist_directory: "./vector_storage"
  collection_name: "concordia_docs"

chunking:
  json_strategy: "hierarchical"
  text_strategy: "semantic"
  chunk_size: 512
  overlap: 50

rag:
  top_k: 5
  similarity_threshold: 0.7
  enable_reranking: false
```

## Target Data Sources
1. `docs/references/openrouter_docs.txt`
2. `concordia/.api_docs/_build/json/` (JSON API documentation)
3. Concordia developer guides and documentation
4. Integration test examples

## Development Status
- **Status**: Phase 1 - Initial setup
- **Next**: Begin ChromaDB integration and document loaders