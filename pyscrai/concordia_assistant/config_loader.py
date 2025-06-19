"""Configuration loader for the Concordia Assistant."""

import os
import yaml
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    provider: str
    sentence_transformers: Dict[str, Any]
    huggingface: Dict[str, Any]


@dataclass
class VectorDBConfig:
    """Configuration for vector database."""
    type: str
    persist_directory: str
    collection_name: str


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    json_strategy: str
    text_strategy: str
    chunk_size: int
    overlap: int


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    top_k: int
    similarity_threshold: float
    enable_reranking: bool


@dataclass
class OpenRouterConfig:
    """Configuration for OpenRouter API."""
    api_key_env: str
    base_url: str
    model_kwargs: Dict[str, Any]


@dataclass
class DataSource:
    """Configuration for a data source."""
    path: str
    type: str


@dataclass
class AssistantConfig:
    """Main configuration for the Concordia Assistant."""
    models: Dict[str, str]
    embedding: EmbeddingConfig
    vector_db: VectorDBConfig
    chunking: ChunkingConfig
    rag: RAGConfig
    openrouter: OpenRouterConfig
    data_sources: List[DataSource]


def load_config(config_path: str = None) -> AssistantConfig:
    """Load configuration from YAML file."""
    if config_path is None:
        # Default to config.yaml in the same directory as this file
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # Convert data sources to DataSource objects
    data_sources = [DataSource(**ds) for ds in config_data['data_sources']]
    
    # Build configuration object
    config = AssistantConfig(
        models=config_data['models'],
        embedding=EmbeddingConfig(**config_data['embedding']),
        vector_db=VectorDBConfig(**config_data['vector_db']),
        chunking=ChunkingConfig(**config_data['chunking']),
        rag=RAGConfig(**config_data['rag']),
        openrouter=OpenRouterConfig(**config_data['openrouter']),
        data_sources=data_sources
    )
    
    return config


def get_hf_home() -> str:
    """Get the HuggingFace home directory from environment or default."""
    return os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface/hub'))


def get_api_key(env_var_name: str) -> Optional[str]:
    """Get API key from environment variable."""
    return os.environ.get(env_var_name)
