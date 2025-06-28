from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve environment variable placeholder."""
    return os.getenv(key, default)


@dataclass
class ModelConfig:
    """Language model configuration."""

    provider: str = "openrouter"
    model: str = "mistralai/mistral-small-3.1-24b-instruct:free"
    model_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            # "temperature": 0.7,
            "max_tokens": 500
        }
    )

    def get_llm(self):
        """Get the configured LLM instance."""
        from ..adapters.llm.factory import create_llm
        return create_llm(self)


@dataclass
class EmbeddingConfig:
    """Embedding model configuration (local sentence-transformers only)."""
    provider: str = "local_sentencetransformers"
    model: str = "all-mpnet-base-v2"
    device: str = "cuda"
    fallback_models: List[str] = field(default_factory=lambda: [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2"
    ])
    cache_folder: Optional[str] = None


@dataclass
class VectorDBConfig:
    """Vector database settings."""

    persist_directory: str = "./vector_storage"
    collection_name: str = " default_docs"
    anonymized_telemetry: bool = False
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "hnsw_space": "cosine"
    })


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    json_strategy: str = "hierarchical"
    text_strategy: str = "semantic"
    chunk_size: int = 512
    overlap: int = 50


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""

    top_k: int = 5
    similarity_threshold: float = 0.7
    enable_reranking: bool = False


@dataclass
class AgentConfig:
    """Main configuration class for RAG agents."""
    models: ModelConfig = field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectordb: VectorDBConfig = field(default_factory=VectorDBConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    system_prompt: str = "You are a helpful RAG developer assistant."
    data_paths: List[str] = field(default_factory=list)
    openrouter_api_key_env: str = "OPENROUTER_API_KEY"


def load_template(name: str = "chat") -> AgentConfig:
    """Load a template configuration by name."""
    from .templates import get_template_path
    return load_config(get_template_path(name))


def list_templates() -> list[str]:
    """List all available configuration templates."""
    from .templates import list_templates
    return list_templates()


def _interpolate_env(data: Any) -> Any:
    """Recursively replace ${VAR} strings with environment variables."""
    if isinstance(data, dict):
        return {k: _interpolate_env(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_interpolate_env(v) for v in data]
    if isinstance(data, str) and data.startswith("${") and data.endswith("}"):
        key = data[2:-1]
        return os.getenv(key, data)
    return data


def load_config(path: str) -> AgentConfig:
    """Load configuration from YAML file."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _interpolate_env(raw)

    return AgentConfig(
        models=ModelConfig(**raw.get("models", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        vectordb=VectorDBConfig(**raw.get("vectordb", {})),
        chunking=ChunkingConfig(**raw.get("chunking", {})),
        rag=RAGConfig(**raw.get("rag", {})),
        system_prompt=raw.get("system_prompt", AgentConfig.system_prompt),
        data_paths=raw.get("data_paths", []),
        openrouter_api_key_env=raw.get("openrouter_api_key_env", AgentConfig.openrouter_api_key_env),
    )
    return AgentConfig(
        models=ModelConfig(**raw.get("models", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        vectordb=VectorDBConfig(**raw.get("vectordb", {})),
        chunking=ChunkingConfig(**raw.get("chunking", {})),
        rag=RAGConfig(**raw.get("rag", {})),
        system_prompt=raw.get("system_prompt", AgentConfig.system_prompt),
        data_paths=raw.get("data_paths", []),
        openrouter_api_key_env=raw.get("openrouter_api_key_env", AgentConfig.openrouter_api_key_env),
    )
