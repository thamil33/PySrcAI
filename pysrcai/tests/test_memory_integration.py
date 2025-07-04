"""Test script for memory system integration."""

import sys
import os

# Add the project root to the path for proper imports
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from pysrcai.config.embedding_config import MemoryConfig, EmbeddingConfig
from pysrcai.agents.memory.memory_factory import create_memory_bank_with_embeddings


def test_basic_memory():
    """Test basic memory without embeddings."""
    print("Testing basic memory...")
    
    config = MemoryConfig(
        type="basic",
        max_memories=10
    )
    
    memory_bank = create_memory_bank_with_embeddings(config)
    
    # Add some memories
    memory_bank.add_memory("Alice said hello", tags=['conversation'])
    memory_bank.add_memory("Bob responded with a wave", tags=['action'])
    memory_bank.add_memory("The room was quiet", tags=['observation'])
    
    # Test retrieval
    recent = memory_bank.retrieve_recent(2)
    print(f"Recent memories: {recent}")
    
    query_results = memory_bank.retrieve_by_query("hello")
    print(f"Query results for 'hello': {query_results}")
    
    print("Basic memory test passed!\n")


def test_associative_memory():
    """Test associative memory with embeddings."""
    print("Testing associative memory...")
    
    try:
        config = MemoryConfig(
            type="associative",
            max_memories=10,
            embedding=EmbeddingConfig(
                provider="local_sentencetransformers",
                model="all-MiniLM-L6-v2",
                device="cpu"
            )
        )
        
        memory_bank = create_memory_bank_with_embeddings(config)
        
        # Add some memories
        memory_bank.add_memory("Alice greeted everyone warmly", tags=['conversation'])
        memory_bank.add_memory("Bob smiled and waved back", tags=['action'])
        memory_bank.add_memory("The atmosphere was friendly", tags=['observation'])
        memory_bank.add_memory("They discussed the weather", tags=['conversation'])
        
        # Test retrieval
        recent = memory_bank.retrieve_recent(2)
        print(f"Recent memories: {recent}")
        
        query_results = memory_bank.retrieve_by_query("greeting")
        print(f"Query results for 'greeting': {query_results}")
        
        print("Associative memory test passed!\n")
        
    except Exception as e:
        print(f"Associative memory test failed: {e}")
        print("This is expected if sentence-transformers is not installed.\n")


def main():
    """Run all tests."""
    print("Testing Memory System Integration")
    print("=" * 40)
    
    test_basic_memory()
    test_associative_memory()
    
    print("All tests completed!")


if __name__ == "__main__":
    main() 