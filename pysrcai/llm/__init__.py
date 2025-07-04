"""Language model system for PySrcAI."""

from .language_model import LanguageModel
from .lmstudio_model import LMStudioLanguageModel
from .openrouter_model import OpenRouterLanguageModel
from .no_language_model import NoLanguageModel
from .retry_wrapper import RetryLanguageModel
from .call_limit_wrapper import CallLimitLanguageModel
from .llm_components import (
    LLMActingComponent,
    ActorLLMComponent,
    ArchonLLMComponent,
    ConfigurableLLMComponent,
)

__all__ = [
    # Base classes
    "LanguageModel",
    
    # Model implementations
    "LMStudioLanguageModel",
    "OpenRouterLanguageModel", 
    "NoLanguageModel",
    
    # Wrappers
    "RetryLanguageModel",
    "CallLimitLanguageModel",
    
    # Components
    "LLMActingComponent",
    "ActorLLMComponent",
    "ArchonLLMComponent",
    "ConfigurableLLMComponent",
]

def create_language_model(model_type: str = "mock"):
    if model_type == "lmstudio":
        return LMStudioLanguageModel(
            model_name="local-model",
            base_url="http://localhost:1234/v1",
            verbose_logging=True
        )
    elif model_type == "openrouter":
        return OpenRouterLanguageModel(
            model_name="mistralai/mistral-7b-instruct:free",
            verbose_logging=True
        )
    else:
        return NoLanguageModel()
