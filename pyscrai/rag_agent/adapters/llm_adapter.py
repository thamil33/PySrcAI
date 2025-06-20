"""Language model adapter for RAG Agents."""

import os
from typing import Optional, Dict, Any

# Import from the `src` namespace where the configuration loader is defined.
from ..config_loader import AgentConfig, get_api_key


class LLMAdapter:
    """Adapter class for language model operations."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.model_name = config.models.language_model
        self.llm = None
        # For config access logging
        self._log_config_access("models.language_model", self.model_name)
        self._initialize_llm()

    def _log_config_access(self, key, value):
        # Use the config access logger if enabled
        try:
            from ..config_access_logger import is_logging_enabled, logger
            if is_logging_enabled():
                logger.info(f"llm_adapter accessed {key} -> {repr(value)}")
        except Exception:
            pass

    def _initialize_llm(self):
        """Initialize the language model using Concordia's OpenRouter integration."""
        try:
            # Import from Concordia's language model module
            from concordia.language_model import openrouter_model

            # Get API key
            api_key = get_api_key("OPENROUTER_API_KEY")

            # Try to get model_kwargs from config if present (e.g., openrouter.model_kwargs)
            model_kwargs = None
            if hasattr(self.config, 'openrouter'):
                model_kwargs = getattr(self.config.openrouter, 'model_kwargs', None)
                self._log_config_access("openrouter.model_kwargs", model_kwargs)
                if model_kwargs and 'max_tokens' in model_kwargs:
                    self._log_config_access("openrouter.model_kwargs.max_tokens", model_kwargs['max_tokens'])

            # Initialize OpenRouter model
            if model_kwargs:
                self.llm = openrouter_model.OpenRouterLanguageModel(
                    api_key=api_key, model_name=self.model_name, **model_kwargs
                )
            else:
                self.llm = openrouter_model.OpenRouterLanguageModel(
                    api_key=api_key, model_name=self.model_name
                )

            print(f"Initialized OpenRouter with model: {self.model_name}")

        except ImportError as e:
            raise ImportError(f"Failed to import Concordia OpenRouter model: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize language model: {e}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the language model."""
        try:
            from ..config_access_logger import is_logging_enabled, logger
            if is_logging_enabled():
                logger.info(f"llm_adapter.generate called with kwargs: {kwargs}")
        except Exception:
            pass
        if hasattr(self.llm, "sample_text"):
            return self.llm.sample_text(prompt, **kwargs)
        else:
            return self.llm(prompt, **kwargs)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {"model_name": self.model_name, "provider": "OpenRouter"}
