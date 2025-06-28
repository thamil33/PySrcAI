"""Agent builder and factory for creating RAG agents."""

from typing import Optional, Dict, Any
from pathlib import Path

from ..config.config import AgentConfig, load_config, load_template
from .rag_agent import RAGAgent
from .chat_agent import ChatAgent
from .base import BaseAgent


class AgentBuilder:
    """Builder/factory for creating and configuring agents (RAG, Chat, etc)."""

    @staticmethod
    def from_config(config: AgentConfig, agent_type: str = "rag") -> BaseAgent:
        """Create an agent from configuration.

        Args:
            config: Agent configuration
            agent_type: Type of agent ("rag" or "chat")
        Returns:
            Configured agent
        """
        if agent_type == "chat":
            return ChatAgent(config)
        return RAGAgent(config)

    @staticmethod
    def from_config_file(config_path: str, agent_type: str = "rag") -> BaseAgent:
        """Create an agent from a configuration file.

        Args:
            config_path: Path to the configuration YAML file
            agent_type: Type of agent ("rag" or "chat")
        Returns:
            Configured agent
        """
        config = load_config(config_path)
        return AgentBuilder.from_config(config, agent_type=agent_type)

    @staticmethod
    def from_template(template_name: str = "default", agent_type: str = "rag") -> BaseAgent:
        """Create an agent from a configuration template.

        Args:
            template_name: Name of the template to use
            agent_type: Type of agent ("rag" or "chat")
        Returns:
            Configured agent
        """
        config = load_template(template_name)
        return AgentBuilder.from_config(config, agent_type=agent_type)

    @staticmethod
    def create_default(agent_type: str = "rag") -> BaseAgent:
        """Create an agent with default configuration.

        Args:
            agent_type: Type of agent ("rag" or "chat")
        Returns:
            Agent with default settings
        """
        config = AgentConfig()
        return AgentBuilder.from_config(config, agent_type=agent_type)

    @staticmethod
    def create_with_overrides(agent_type: str = "rag", **overrides) -> BaseAgent:
        """Create an agent with configuration overrides.

        Args:
            agent_type: Type of agent ("rag" or "chat")
            **overrides: Configuration overrides
        Returns:
            Configured agent
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
        return AgentBuilder.from_config(config, agent_type=agent_type)

    @staticmethod
    def quick_setup(
        data_paths: Optional[list] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        collection_name: Optional[str] = None,
        agent_type: str = "rag",
        **kwargs
    ) -> BaseAgent:
        """Quick setup for an agent with common overrides.

        Args:
            data_paths: List of data paths to ingest
            system_prompt: Custom system prompt
            model: LLM model to use
            embedding_model: Embedding model to use
            collection_name: Vector store collection name
            agent_type: Type of agent ("rag" or "chat")
            **kwargs: Additional configuration overrides
        Returns:
            Configured agent
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
        return AgentBuilder.from_config(config, agent_type=agent_type)


def create_agent(
    config_source: Optional[str] = None,
    template: Optional[str] = None,
    agent_type: str = "rag",
    **overrides
) -> BaseAgent:
    """Convenience function to create an agent (RAG or Chat).

    Args:
        config_source: Path to config file, or None for default
        template: Template name to use, or None for default
        agent_type: Type of agent ("rag" or "chat")
        **overrides: Configuration overrides
    Returns:
        Configured agent
    """
    if config_source:
        if Path(config_source).exists():
            agent = AgentBuilder.from_config_file(config_source, agent_type=agent_type)
        else:
            raise FileNotFoundError(f"Config file not found: {config_source}")
    elif template:
        agent = AgentBuilder.from_template(template, agent_type=agent_type)
    else:
        agent = AgentBuilder.create_default(agent_type=agent_type)
    # Apply any overrides
    if overrides:
        for key, value in overrides.items():
            if hasattr(agent.config, key):
                setattr(agent.config, key, value)
    return agent
