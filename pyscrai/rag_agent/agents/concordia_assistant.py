"""Concordia Assistant - specialized RAG agent for Concordia framework documentation."""

from typing import List
from ..base_rag_agent import BaseRAGAgent
from ..config_loader import AgentConfig


class ConcordiaAssistant(BaseRAGAgent):
    """
    Specialized RAG agent for answering questions about the Concordia framework.
    
    Provides expertise in:
    - Concordia component architecture
    - Agent development patterns
    - Integration examples
    - API documentation
    """
    
    def get_system_prompt(self) -> str:
        """Return the system prompt for Concordia assistance."""
        return """You are an expert assistant for the Concordia framework, a Python library for building agent-based simulations.

Your expertise includes:
- Concordia's component-based architecture
- Entity-agent patterns and implementations
- Language model integration via OpenRouter
- Simulation design and development
- Memory systems and associative memory banks
- Game master and environment components
- Testing and validation approaches

When answering questions:
1. Provide accurate, practical guidance based on the Concordia documentation
2. Include code examples when helpful
3. Explain the reasoning behind architectural decisions
4. Suggest best practices for simulation development
5. Reference specific Concordia modules and classes when relevant

Always base your responses on the provided context from the Concordia documentation and codebase."""

    def get_agent_name(self) -> str:
        """Return the name of this agent."""
        return "ConcordiaAssistant"
    
    def get_default_data_sources(self) -> List[str]:
        """Return default data sources for Concordia documentation."""
        return [
            "docs/concordia/",
            "concordia/.api_docs/_build/json/",
            "docs/concordia_developers_guide.md",
            "docs/concordia_overview.md",
            "concordia/concordia_integration_test.py",
            "concordia/concordia_integration_test.md",
        ]
