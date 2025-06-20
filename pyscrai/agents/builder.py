"""Agent builder and factory for creating RAG agents."""

from typing import Optional, Dict, Any
from pathlib import Path

from ..config.config import AgentConfig, load_config, load_template
from .rag_agent import RAGAgent
from .base import BaseAgent


class AgentBuilder:
    """Builder/factory for creating and configuring RAG agents."""
    
    @staticmethod
    def from_config(config: AgentConfig) -> RAGAgent:
        """Create a RAG agent from configuration.
        
        Args:
            config: Agent configuration
            
        Returns:
            Configured RAG agent
        """
        return RAGAgent(config)
    
    @staticmethod
    def from_config_file(config_path: str) -> RAGAgent:
        """Create a RAG agent from a configuration file.
        
        Args:
            config_path: Path to the configuration YAML file
            
        Returns:
            Configured RAG agent
        """
        config = load_config(config_path)
        return AgentBuilder.from_config(config)
    
    @staticmethod
    def from_template(template_name: str = "default") -> RAGAgent:
        """Create a RAG agent from a configuration template.
        
        Args:
            template_name: Name of the template to use
            
        Returns:
            Configured RAG agent
        """
        config = load_template(template_name)
        return AgentBuilder.from_config(config)
    
    @staticmethod
    def create_default() -> RAGAgent:
        """Create a RAG agent with default configuration.
        
        Returns:
            RAG agent with default settings
        """
        config = AgentConfig()
        return AgentBuilder.from_config(config)
    
    @staticmethod
    def create_with_overrides(**overrides) -> RAGAgent:
        """Create a RAG agent with configuration overrides.
        
        Args:
            **overrides: Configuration overrides
            
        Returns:
            Configured RAG agent
        """
        config = AgentConfig()
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                # Handle nested config updates
                if '.' in key:
                    obj, attr = key.rsplit('.', 1)
                    nested_obj = getattr(config, obj)
                    if hasattr(nested_obj, attr):
                        setattr(nested_obj, attr, value)
        
        return AgentBuilder.from_config(config)
    
    @staticmethod
    def quick_setup(
        data_paths: Optional[list] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        collection_name: Optional[str] = None,
        **kwargs
    ) -> RAGAgent:
        """Quick setup for a RAG agent with common overrides.
        
        Args:
            data_paths: List of data paths to ingest
            system_prompt: Custom system prompt
            model: LLM model to use
            embedding_model: Embedding model to use
            collection_name: Vector store collection name
            **kwargs: Additional configuration overrides
            
        Returns:
            Configured RAG agent
        """
        config = AgentConfig()
        
        if data_paths:
            config.data_paths = data_paths
        
        if system_prompt:
            config.system_prompt = system_prompt
        
        if model:
            config.models.model = model
        
        if embedding_model:
            config.embedding.model = embedding_model
        
        if collection_name:
            config.vectordb.collection_name = collection_name
        
        # Apply additional overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return AgentBuilder.from_config(config)


def create_agent(
    config_source: Optional[str] = None,
    template: Optional[str] = None,
    **overrides
) -> BaseAgent:
    """Convenience function to create a RAG agent.
    
    Args:
        config_source: Path to config file, or None for default
        template: Template name to use, or None for default
        **overrides: Configuration overrides
        
    Returns:
        Configured RAG agent
    """
    if config_source:
        if Path(config_source).exists():
            agent = AgentBuilder.from_config_file(config_source)
        else:
            raise FileNotFoundError(f"Config file not found: {config_source}")
    elif template:
        agent = AgentBuilder.from_template(template)
    else:
        agent = AgentBuilder.create_default()
    
    # Apply any overrides
    if overrides:
        for key, value in overrides.items():
            if hasattr(agent.config, key):
                setattr(agent.config, key, value)
    
    return agent
