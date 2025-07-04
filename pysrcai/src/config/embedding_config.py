"""Embedding configuration classes for PySrcAI."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    
    provider: str = "local_sentencetransformers"
    model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    cache_folder: Optional[str] = None
    fallback_models: List[str] = field(default_factory=list)
    
    # Additional settings
    normalize_embeddings: bool = True
    batch_size: int = 32


@dataclass
class VectorDBConfig:
    """Configuration for vector databases."""
    
    provider: str = "chroma"
    collection_name: str = "pysrcai_memories"
    persist_directory: str = "./data/vectordb"
    settings: Dict[str, Any] = field(default_factory=dict)
    
    # Chroma-specific settings
    collection_metadata: Optional[Dict[str, Any]] = None
    distance_function: str = "cosine"


@dataclass
class MemoryConfig:
    """Configuration for memory systems."""
    
    type: str = "associative"  # "basic" or "associative"
    max_memories: int = 1000
    importance_threshold: float = 0.5
    max_context_memories: int = 5
    
    # Embedding settings for associative memory
    embedding: Optional[EmbeddingConfig] = None
    
    # Vector store settings for persistence
    vectorstore: Optional[VectorDBConfig] = None 