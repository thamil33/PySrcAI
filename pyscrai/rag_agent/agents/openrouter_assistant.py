"""OpenRouter Assistant - specialized RAG agent for OpenRouter API documentation."""

from typing import List

# Import from src namespace to locate implementation correctly
from ..src.base_rag_agent import BaseRAGAgent
from ..src.config_loader import AgentConfig


class OpenRouterAssistant(BaseRAGAgent):
    """
    Specialized RAG agent for answering questions about the OpenRouter API.

    Provides expertise in:
    - OpenRouter API endpoints and usage
    - Model selection and capabilities
    - Authentication and API keys
    - Rate limits and best practices
    - Integration patterns
    """

    def get_system_prompt(self) -> str:
        """Return the system prompt for OpenRouter assistance."""
        return """You are an expert assistant for the OpenRouter API, a unified interface for accessing multiple AI models.

Your expertise includes:
- OpenRouter API endpoints and authentication
- Model selection, capabilities, and pricing
- Request formatting and response handling
- Rate limiting and error handling
- Integration best practices
- Model comparison and recommendations

When answering questions:
1. Provide accurate information about OpenRouter's API and models
2. Include practical code examples for API usage
3. Explain model capabilities and use cases
4. Suggest appropriate models for different tasks
5. Address common integration challenges
6. Reference specific API endpoints and parameters

Always base your responses on the provided OpenRouter documentation and maintain accuracy about current API features and model availability."""

    def get_agent_name(self) -> str:
        """Return the name of this agent."""
        return "OpenRouterAssistant"

    def get_default_data_sources(self) -> List[str]:
        """Return default data sources for OpenRouter documentation."""
        return [
            "docs/references/openrouter_docs.txt",
            "concordia/language_model/openrouter_model.py",
        ]
