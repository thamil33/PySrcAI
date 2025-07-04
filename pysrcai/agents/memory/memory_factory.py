"""Memory factory for creating memory banks with embedding support."""

from typing import Optional, Callable, Any
from collections.abc import Sequence

from .memory_components import MemoryBank, BasicMemoryBank, AssociativeMemoryBank
from ...config.embedding_config import MemoryConfig, EmbeddingConfig, VectorDBConfig


def create_memory_bank(
    config: MemoryConfig,
    embedder: Optional[Callable[[str], Any]] = None
) -> MemoryBank:
    """Create a memory bank based on configuration.
    
    Args:
        config: Memory configuration specifying type and settings
        embedder: Optional embedder function for associative memory
        
    Returns:
        Configured memory bank instance
        
    Raises:
        ValueError: If configuration is invalid
    """
    if config.type == "basic":
        return BasicMemoryBank(max_memories=config.max_memories)
    
    elif config.type == "associative":
        if embedder is None:
            raise ValueError("Associative memory requires an embedder function")
        
        memory_bank = AssociativeMemoryBank(
            embedder=embedder,
            max_memories=config.max_memories
        )
        return memory_bank
    
    else:
        raise ValueError(f"Unsupported memory type: {config.type}")


def create_embedder_from_config(config: EmbeddingConfig) -> Callable[[str], Any]:
    """Create an embedder function from configuration.
    
    Args:
        config: Embedding configuration
        
    Returns:
        Embedder function that takes text and returns embeddings
    """
    # Import here to avoid circular imports
    from ...embeddings.factory import create_embedder
    
    embedder_instance = create_embedder(config)
    
    def embed_text(text: str) -> Any:
        """Embed a single text string."""
        return embedder_instance.embed_query(text)
    
    return embed_text


def create_memory_bank_with_embeddings(config: MemoryConfig) -> MemoryBank:
    """Create a memory bank with automatic embedding setup.
    
    Args:
        config: Memory configuration with embedding settings
        
    Returns:
        Configured memory bank with embeddings
    """
    embedder = None
    if config.embedding and config.type == "associative":
        embedder = create_embedder_from_config(config.embedding)
    
    return create_memory_bank(config, embedder) 