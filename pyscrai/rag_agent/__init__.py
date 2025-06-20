"""
Base RAG Agent Framework

Provides a flexible, extensible RAG agent system with builder pattern support
for creating specialized assistants with custom configurations, instructions,
and data sources.
"""

from .src.base_rag_agent import BaseRAGAgent
from .src.rag_agent_builder import RAGAgentBuilder, create_agent, quick_agent
from .config_loader import load_config, AgentConfig
from .adapters import EmbeddingAdapter, VectorDBAdapter, LLMAdapter

__version__ = "0.2.0"
__all__ = [
    "BaseRAGAgent", 
    "RAGAgentBuilder",
    "create_agent",
    "quick_agent",
    "load_config", 
    "AgentConfig",
    "EmbeddingAdapter",
    "VectorDBAdapter", 
    "LLMAdapter"
]
