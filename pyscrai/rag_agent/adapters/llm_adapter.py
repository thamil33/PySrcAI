"""Language model adapter for RAG Agents."""

import os
from typing import Optional, Dict, Any
from ..config_loader import AgentConfig, get_api_key


class LLMAdapter:
    """Adapter class for language model operations."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.model_name = config.models.language_model
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the language model using Concordia's OpenRouter integration."""
        try:
            # Import from Concordia's language model module
            from concordia.language_model import openrouter_model
            
            # Get API key
            api_key = get_api_key("OPENROUTER_API_KEY")
            
            # Initialize OpenRouter model
            self.llm = openrouter_model.OpenRouterLanguageModel(
                api_key=api_key,
                model_name=self.model_name
            )
            
            print(f"Initialized OpenRouter with model: {self.model_name}")
            
        except ImportError as e:
            raise ImportError(f"Failed to import Concordia OpenRouter model: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize language model: {e}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the language model."""
        try:
            response = self.llm(prompt, **kwargs)
            return response
        except Exception as e:
            raise RuntimeError(f"Error generating response: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'provider': 'OpenRouter'
        }
