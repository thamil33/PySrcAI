"""Configuration loader for RAG Agents."""

import os
import yaml
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class ModelsConfig:
    """Configuration for models (LLM and embedding)."""
    language_model: str = "mistralai/mistral-small-3.1-24b-instruct:free"
    embedding_provider: str = "huggingface_api"
    embedding_model: str = "BAAI/bge-base-en-v1.5"


@dataclass
class VectorDBConfig:
    """Configuration for vector database."""
    type: str = "chromadb"
    persist_directory: str = "./vector_storage"
    collection_name: str = "rag_agent_docs"


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
    models: ModelsConfig
    vector_db: VectorDBConfig
    chunking: ChunkingConfig
    rag: RAGConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """Create AgentConfig from dictionary."""
        return cls(
            models=ModelsConfig(**config_dict.get('models', {})),
            vector_db=VectorDBConfig(**config_dict.get('vector_db', {})),
            chunking=ChunkingConfig(**config_dict.get('chunking', {})),
            rag=RAGConfig(**config_dict.get('rag', {}))
        )


def load_config(config_path: str = None) -> AgentConfig:
    """Load configuration from YAML file."""
    if config_path is None:
        # Default to config.yaml in the same directory as this file
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    
    # Check if file exists
    if not os.path.exists(config_path):
        # Return default configuration if file doesn't exist
        return _create_default_config()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    return AgentConfig.from_dict(config_data)


def _create_default_config() -> AgentConfig:
    """Create a default configuration."""
    default_config = {
        'models': {
            'language_model': 'mistralai/mistral-small-3.1-24b-instruct:free',
            'embedding_provider': 'huggingface_api',
            'embedding_model': 'BAAI/bge-base-en-v1.5'
        },
        'vector_db': {
            'type': 'chromadb',
            'persist_directory': './vector_storage',
            'collection_name': 'rag_agent_docs'
        },
        'chunking': {
            'json_strategy': 'hierarchical',
            'text_strategy': 'semantic',
            'chunk_size': 512,
            'overlap': 50
        },
        'rag': {
            'top_k': 5,
            'similarity_threshold': 0.7,
            'enable_reranking': False
        }
    }
    
    return AgentConfig.from_dict(default_config)


def get_hf_home() -> str:
    """Get the HuggingFace cache directory."""
    return os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def get_api_key(env_var: str) -> str:
    """Get API key from environment variable."""
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"Environment variable {env_var} not set")
    return api_key
