"""PySrcAI Memory Module.

This module provides memory capabilities for agents:
- BasicMemoryBank: Simple chronological memory storage
- AssociativeMemoryBank: Advanced embedding-based memory retrieval
- MemoryComponent: Context component for memory integration
- Simple embedders for text similarity

Usage Example:
    from pysrcai.src.agents.memory import BasicMemoryBank, MemoryComponent
    
    # Create a memory bank
    memory_bank = BasicMemoryBank(max_memories=500)
    
    # Create memory component for an agent
    memory_component = MemoryComponent(memory_bank)
    
    # Add to agent's context components
    context_components = {"memory": memory_component}
"""

from .memory_components import (
    MemoryBank,
    BasicMemoryBank,
    AssociativeMemoryBank,
    MemoryComponent,
)

from .embedders import (
    SimpleEmbedder,
    HashEmbedder,
    create_simple_embedder,
    create_hash_embedder,
)

__all__ = [
    # Memory banks
    "MemoryBank",
    "BasicMemoryBank", 
    "AssociativeMemoryBank",
    
    # Components
    "MemoryComponent",
    
    # Embedders
    "SimpleEmbedder",
    "HashEmbedder",
    "create_simple_embedder",
    "create_hash_embedder",
]
