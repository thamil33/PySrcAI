"""RAG Agent Builder - provides a fluent interface for creating custom RAG agents."""

import os
from typing import List, Dict, Any, Optional, Type, Callable
from .base_rag_agent import BaseRAGAgent
from ..config_loader import AgentConfig, load_config


class CustomRAGAgent(BaseRAGAgent):
    """A customizable RAG agent created via the builder pattern."""
    
    def __init__(self, config: AgentConfig, system_prompt: str, agent_name: str, 
                 data_sources: List[str]):
        super().__init__(config)
        self._system_prompt = system_prompt
        self._agent_name = agent_name
        self._data_sources = data_sources
    
    def get_system_prompt(self) -> str:
        return self._system_prompt
    
    def get_agent_name(self) -> str:
        return self._agent_name
    
    def get_default_data_sources(self) -> List[str]:
        return self._data_sources


class RAGAgentBuilder:
    """
    Builder class for creating custom RAG agents with fluent interface.
    
    Example usage:
        agent = (RAGAgentBuilder()
                .with_name("MyCustomAgent")
                .with_system_prompt("You are a helpful assistant...")
                .with_data_sources(["docs/", "knowledge_base.txt"])
                .with_config_file("custom_config.yaml")
                .build())
    """
    
    def __init__(self):
        self._config_path: Optional[str] = None
        self._config: Optional[AgentConfig] = None
        self._system_prompt: Optional[str] = None
        self._agent_name: str = "CustomRAGAgent"
        self._data_sources: List[str] = []
        self._embedding_provider: Optional[str] = None
        self._llm_model: Optional[str] = None
        self._vector_db_settings: Dict[str, Any] = {}
    
    def with_config_file(self, config_path: str) -> 'RAGAgentBuilder':
        """Load configuration from a YAML file."""
        self._config_path = config_path
        return self
    
    def with_config(self, config: AgentConfig) -> 'RAGAgentBuilder':
        """Use a pre-loaded configuration object."""
        self._config = config
        return self
    
    def with_name(self, name: str) -> 'RAGAgentBuilder':
        """Set the agent name."""
        self._agent_name = name
        return self
    
    def with_system_prompt(self, prompt: str) -> 'RAGAgentBuilder':
        """Set the system prompt for the agent."""
        self._system_prompt = prompt
        return self
    
    def with_data_sources(self, sources: List[str]) -> 'RAGAgentBuilder':
        """Set the default data sources for the agent."""
        self._data_sources = sources
        return self
    
    def with_embedding_provider(self, provider: str) -> 'RAGAgentBuilder':
        """Set the embedding provider (e.g., 'huggingface_api', 'local_sentencetransformers')."""
        self._embedding_provider = provider
        return self
    
    def with_llm_model(self, model: str) -> 'RAGAgentBuilder':
        """Set the language model to use."""
        self._llm_model = model
        return self
    
    def with_vector_db_collection(self, collection_name: str) -> 'RAGAgentBuilder':
        """Set the vector database collection name."""
        self._vector_db_settings['collection_name'] = collection_name
        return self
    
    def with_vector_db_path(self, persist_directory: str) -> 'RAGAgentBuilder':
        """Set the vector database persistence directory."""
        self._vector_db_settings['persist_directory'] = persist_directory
        return self
    
    def with_rag_settings(self, top_k: int = None, similarity_threshold: float = None) -> 'RAGAgentBuilder':
        """Configure RAG retrieval settings."""
        if top_k is not None:
            self._vector_db_settings['top_k'] = top_k
        if similarity_threshold is not None:
            self._vector_db_settings['similarity_threshold'] = similarity_threshold
        return self
    
    def build(self) -> CustomRAGAgent:
        """Build and return the configured RAG agent."""
        # Load or use provided configuration
        if self._config:
            config = self._config
        elif self._config_path:
            config = load_config(self._config_path)
        else:
            # Default to templates/concordia.yaml
            default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rag_agent", "templates", "concordia.yaml")
            config = load_config(default_path)

        # Apply any builder-specific overrides
        config = self._apply_overrides(config)

        # Use agent section from config if present
        agent_section = getattr(config, 'agent', None)
        system_prompt = self._system_prompt or (agent_section.system_prompt if agent_section and hasattr(agent_section, 'system_prompt') else None)
        agent_name = self._agent_name or (agent_section.name if agent_section and hasattr(agent_section, 'name') else "CustomRAGAgent")
        data_sources = self._data_sources or (agent_section.data_sources if agent_section and hasattr(agent_section, 'data_sources') else [])

        if not system_prompt:
            raise ValueError("System prompt is required. Use with_system_prompt() or provide it in the config agent section.")

        return CustomRAGAgent(
            config=config,
            system_prompt=system_prompt,
            agent_name=agent_name,
            data_sources=data_sources
        )
    
    def _create_default_config(self) -> AgentConfig:
        """Create a default configuration."""
        from ..config_loader import AgentConfig
        
        # Create basic default config structure
        default_config = {
            'models': {
                'language_model': 'mistralai/mistral-small-3.1-24b-instruct:free',
                'embedding_provider': 'huggingface_api',
                'embedding_model': 'BAAI/bge-base-en-v1.5'
            },
            'vector_db': {
                'type': 'chromadb',
                'persist_directory': './vector_storage',
                'collection_name': f"{self._agent_name.lower()}_docs"
            },
            'chunking': {
                'json_strategy': 'hierarchical',
                'text_strategy': 'semantic',
                'chunk_size': 512,
                'overlap': 50
            },
            'rag': {
                'top_k': 5,
                'similarity_threshold': 0.7,
                'enable_reranking': False
            }
        }
        
        return AgentConfig.from_dict(default_config)
    
    def _apply_overrides(self, config: AgentConfig) -> AgentConfig:
        """Apply builder-specific overrides to the configuration."""
        # Override embedding provider if specified
        if self._embedding_provider:
            config.models.embedding_provider = self._embedding_provider
        
        # Override LLM model if specified
        if self._llm_model:
            config.models.language_model = self._llm_model
        
        # Apply vector DB settings
        for key, value in self._vector_db_settings.items():
            if hasattr(config.vector_db, key):
                setattr(config.vector_db, key, value)
            elif hasattr(config.rag, key):
                setattr(config.rag, key, value)
        
        return config




def create_agent(config_file: str = None, **kwargs) -> BaseRAGAgent:
    """
    Factory function for creating RAG agents based on config file contents.
    If name/system_prompt/data_sources are provided, use the builder for a custom agent.
    Otherwise, use the agent section in the config or default to templates/concordia.yaml.
    """
    builder = RAGAgentBuilder()
    if 'name' in kwargs:
        builder.with_name(kwargs['name'])
    if 'system_prompt' in kwargs:
        builder.with_system_prompt(kwargs['system_prompt'])
    if 'data_sources' in kwargs:
        builder.with_data_sources(kwargs['data_sources'])
    if config_file:
        builder.with_config_file(config_file)
    if 'embedding_provider' in kwargs:
        builder.with_embedding_provider(kwargs['embedding_provider'])
    if 'llm_model' in kwargs:
        builder.with_llm_model(kwargs['llm_model'])
    return builder.build()


# Template function for quick agent creation
def quick_agent(name: str, system_prompt: str, data_sources: List[str], 
                config_file: str = None) -> CustomRAGAgent:
    """
    Quick helper function for creating a basic RAG agent.
    
    Args:
        name: Agent name
        system_prompt: System prompt for the agent
        data_sources: List of data source paths
        config_file: Optional config file path
    
    Returns:
        Configured RAG agent
    """
    builder = (RAGAgentBuilder()
               .with_name(name)
               .with_system_prompt(system_prompt)
               .with_data_sources(data_sources))
    
    if config_file:
        builder.with_config_file(config_file)
    
    return builder.build()
