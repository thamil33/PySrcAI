#!/usr/bin/env python3
"""
Test script to verify that basic imports work after refactoring.
Run this script to check that the main components can be imported successfully.
"""

def test_main_imports():
    """Test that main package imports work."""
    print("Testing main package imports...")
    
    try:
        from pysrcai import SimulationFactory, SimulationEngine, SequentialEngine
        from pysrcai import Agent, Actor, Archon
        from pysrcai import LanguageModel
        print("✅ Main package imports successful")
    except ImportError as e:
        print(f"❌ Main package import failed: {e}")
        return False
    
    return True

def test_agent_imports():
    """Test that agent imports work."""
    print("Testing agent imports...")
    
    try:
        from pysrcai.agents.base import Agent, Actor, Archon, AgentFactory
        from pysrcai.agents.memory import BasicMemoryBank, MemoryComponent
        from pysrcai.agents.components import ComponentFactory
        print("✅ Agent imports successful")
    except ImportError as e:
        print(f"❌ Agent import failed: {e}")
        return False
    
    return True

def test_llm_imports():
    """Test that LLM imports work."""
    print("Testing LLM imports...")
    
    try:
        from pysrcai.llm import LanguageModel, LMStudioLanguageModel, OpenRouterLanguageModel
        from pysrcai.llm import RetryLanguageModel, CallLimitLanguageModel
        print("✅ LLM imports successful")
    except ImportError as e:
        print(f"❌ LLM import failed: {e}")
        return False
    
    return True

def test_embedding_imports():
    """Test that embedding imports work."""
    print("Testing embedding imports...")
    
    try:
        from pysrcai.embeddings import BaseEmbedder, create_embedder
        from pysrcai.embeddings import SentenceTransformerEmbeddings
        print("✅ Embedding imports successful")
    except ImportError as e:
        print(f"❌ Embedding import failed: {e}")
        return False
    
    return True

def test_core_imports():
    """Test that core imports work."""
    print("Testing core imports...")
    
    try:
        from pysrcai.core import SimulationFactory, SimulationEngine, SequentialEngine
        print("✅ Core imports successful")
    except ImportError as e:
        print(f"❌ Core import failed: {e}")
        return False
    
    return True

def test_utils_imports():
    """Test that utility imports work."""
    print("Testing utility imports...")
    
    try:
        from pysrcai.utils import concurrency, measurements, sampling, text
        print("✅ Utility imports successful")
    except ImportError as e:
        print(f"❌ Utility import failed: {e}")
        return False
    
    return True

def main():
    """Run all import tests."""
    print("PySrcAI Import Test")
    print("=" * 50)
    
    tests = [
        test_main_imports,
        test_agent_imports,
        test_llm_imports,
        test_embedding_imports,
        test_core_imports,
        test_utils_imports,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All imports working correctly!")
        return 0
    else:
        print("⚠️  Some imports failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main()) 