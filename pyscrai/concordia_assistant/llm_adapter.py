"""OpenRouter language model adapter for the Concordia Assistant."""

import os
import sys
from typing import Any, Dict, Optional
from .config_loader import AssistantConfig, get_api_key


class OpenRouterAdapter:
    """Adapter for OpenRouter language model using Concordia's implementation."""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the OpenRouter model using Concordia's implementation."""
        try:
            # Add the parent directory to sys.path to import concordia modules
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            
            from concordia.language_model.openrouter_model import OpenRouterLanguageModel
            
            # Get API key from environment
            api_key = get_api_key(self.config.openrouter.api_key_env)
            if not api_key:
                raise ValueError(f"API key not found in environment variable: {self.config.openrouter.api_key_env}")
            
            # Initialize the model
            self.model = OpenRouterLanguageModel(
                model_name=self.config.models["language_model"],
                api_key=api_key,
                **self.config.openrouter.model_kwargs
            )
            
            print(f"Initialized OpenRouter model: {self.config.models['language_model']}")
            
        except ImportError as e:
            raise ImportError(f"Failed to import OpenRouter model from Concordia: {e}")
        except Exception as e:
            raise Exception(f"Failed to initialize OpenRouter model: {e}")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the language model."""
        try:
            # Merge any additional kwargs with default model kwargs
            generation_kwargs = {**self.config.openrouter.model_kwargs, **kwargs}
            
            # Use the Concordia model's interface
            response = self.model.sample_text(prompt, **generation_kwargs)
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: Unable to generate response - {str(e)}"
    
    def generate_chat_response(self, messages: list, **kwargs) -> str:
        """Generate a response from a chat-style conversation."""
        # Convert messages to a single prompt for the model
        prompt = self._format_chat_messages(messages)
        return self.generate_response(prompt, **kwargs)
    
    def _format_chat_messages(self, messages: list) -> str:
        """Convert chat messages to a single prompt string."""
        formatted_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
            else:
                formatted_parts.append(f"{role}: {content}")
        
        return "\n\n".join(formatted_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_name": self.config.models["language_model"],
            "api_base": self.config.openrouter.base_url,
            "model_kwargs": self.config.openrouter.model_kwargs
        }
