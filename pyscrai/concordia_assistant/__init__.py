"""
Concordia RAG Assistant

A RAG-powered AI assistant using Concordia framework components to provide
intelligent responses about OpenRouter API and Concordia development.
"""

from .rag_pipeline import ConcordiaRAGPipeline
from .config_loader import load_config, AssistantConfig
from .cli import ConcordiaAssistantCLI

__version__ = "0.1.0"
__all__ = ["ConcordiaRAGPipeline", "load_config", "AssistantConfig", "ConcordiaAssistantCLI"]
