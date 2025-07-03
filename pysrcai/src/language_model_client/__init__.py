"""PySrcAI Language Model Client Module.

This module provides language model integrations for PySrcAI agents.
It includes support for:
- LM Studio (local models)
- OpenRouter (cloud models)
- Call limiting and retry wrappers
- No-op model for testing

Usage Example:
    from pysrcai.src.language_model_client import LMStudioLanguageModel, OpenRouterLanguageModel
    
    # Local model via LM Studio
    local_model = LMStudioLanguageModel(
        model_name="my-local-model",
        base_url="http://localhost:1234/v1"
    )
    
    # Cloud model via OpenRouter
    cloud_model = OpenRouterLanguageModel(
        model_name="mistralai/mistral-small-3.1-24b-instruct:free",
        api_key="your-openrouter-key"
    )
"""

from .language_model import LanguageModel, InvalidResponseError
from .lmstudio_model import LMStudioLanguageModel  
from .openrouter_model import OpenRouterLanguageModel
from .no_language_model import NoLanguageModel
from .call_limit_wrapper import CallLimitLanguageModel
from .retry_wrapper import RetryLanguageModel

__all__ = [
    # Base classes
    "LanguageModel",
    "InvalidResponseError",
    
    # Model implementations
    "LMStudioLanguageModel",
    "OpenRouterLanguageModel", 
    "NoLanguageModel",
    
    # Wrappers
    "CallLimitLanguageModel",
    "RetryLanguageModel",
]
