# Add dataclass for the agent section
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class AgentSection:
    name: str = "CustomRAGAgent"
    system_prompt: Optional[str] = None
    data_sources: List[Dict[str, Any]] = field(default_factory=list)
"""Configuration loader for RAG Agents."""


import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

# Import config access logger
try:
    from .config_access_logger import ConfigAccessLogger, is_logging_enabled
except ImportError:
    ConfigAccessLogger = None
    def is_logging_enabled():
        return False


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""
    provider: str = "local_sentencetransformers"
    model: str = "sentence-transformers/all-mpnet-base-v2"
    device: str = "cuda"
    local_files_only: bool = False
    cache_folder: Optional[str] = None  # Will default to HF_HOME
    fallback_models: List[str] = field(default_factory=lambda: [
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-small-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "intfloat/e5-base-v2"
    ])


@dataclass
class ModelsConfig:
    """Configuration for language models."""
    language_model: str = "mistralai/mistral-small-3.1-24b-instruct:free"


@dataclass
class VectorDBConfig:
    """Configuration for vector database."""
    type: str = "chromadb"
    persist_directory: str = "./vector_storage"
    collection_name: str = "rag_agent_docs"
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "anonymized_telemetry": False,
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
    models: ModelsConfig
    embedding: EmbeddingConfig
    vector_db: VectorDBConfig
    chunking: ChunkingConfig
    rag: RAGConfig
    agent: Optional[AgentSection] = None
    openrouter: Optional[dict] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        agent_section = config_dict.get('agent')
        agent = AgentSection(**agent_section) if agent_section else None
        return cls(
            models=ModelsConfig(**config_dict.get('models', {})),
            embedding=EmbeddingConfig(**config_dict.get('embedding', {})),
            vector_db=VectorDBConfig(**config_dict.get('vector_db', {})),
            chunking=ChunkingConfig(**config_dict.get('chunking', {})),
            rag=RAGConfig(**config_dict.get('rag', {})),
            agent=agent,
            openrouter=config_dict.get('openrouter')
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

    config = AgentConfig.from_dict(config_data)
    # Optionally wrap with config access logger
    if is_logging_enabled() and ConfigAccessLogger is not None:
        return ConfigAccessLogger(config)
    return config


def _create_default_config() -> AgentConfig:
    """Create a default configuration."""
    default_config = {
        'models': {
            'language_model': 'mistralai/mistral-small-3.1-24b-instruct:free'
        },
        'embedding': {
            'provider': 'huggingface_api',
            'model': 'BAAI/bge-base-en-v1.5',
            'device': 'cuda',
            'local_files_only': False
        },
        'vector_db': {
            'type': 'chromadb',
            'persist_directory': './vector_storage',
            'collection_name': 'rag_agent_docs',
            'settings': {
                'anonymized_telemetry': False,
                'hnsw_space': 'cosine'
            }
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
    
    config = AgentConfig.from_dict(default_config)
    if is_logging_enabled() and ConfigAccessLogger is not None:
        return ConfigAccessLogger(config)
    return config


def get_hf_home() -> str:
    """Get the HuggingFace cache directory."""
    return os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def get_api_key(env_var: str) -> str:
    """Get API key from environment variable."""
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"Environment variable {env_var} not set")
    return api_key
