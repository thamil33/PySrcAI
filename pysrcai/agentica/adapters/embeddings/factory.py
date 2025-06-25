"""Factory for creating embedding adapter."""

from typing import Type

from ...config.config import EmbeddingConfig
from .base import BaseEmbedder

from .sentence_transformers import SentenceTransformerEmbeddings


def create_embedder(config: EmbeddingConfig) -> BaseEmbedder:
    """Create an embedding adapter based on configuration.
    
    Args:
        config: Embedding configuration specifying provider and settings
        
    Returns:
        Configured embedding adapter instance
        
    Raises:
        ValueError: If provider is not supported
    """
    providers = {
        "local_sentencetransformers": SentenceTransformerEmbeddings
    }
    provider_class = providers.get(config.provider)
    if not provider_class:
        raise ValueError(
            f"Unsupported embedding provider: {config.provider}. "
            f"Must be one of: {', '.join(providers.keys())}"
        )
    return provider_class(config)
