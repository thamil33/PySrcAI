"""Factory for creating LLM instances."""

from typing import Union
from ...config.config import ModelConfig
from .openrouter_adapter import OpenRouterLLM
from .lmstudio_adapter import LMStudioLLM


def create_llm(config: ModelConfig) -> Union[OpenRouterLLM, LMStudioLLM]:
    """Create an LLM instance based on configuration.

    Args:
        config: Model configuration

    Returns:
        LLM instance

    Raises:
        ValueError: If provider is not supported
    """
    if config.provider == "openrouter":
        return OpenRouterLLM(
            model=config.model,
            **config.model_kwargs
        )
    elif config.provider == "lmstudio":
        return LMStudioLLM(
            model=config.model,
            **config.model_kwargs
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
