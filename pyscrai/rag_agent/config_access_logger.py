"""Config access logger and proxy for RAG agent config debugging."""

import logging
import os
from typing import Any

LOG_ENV_VAR = "RAG_AGENT_LOG_CONFIG_ACCESS"

# Set up a logger for config access
logger = logging.getLogger("rag_agent.config_access")
handler = logging.StreamHandler()
formatter = logging.Formatter("[CONFIG ACCESS] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

def is_logging_enabled() -> bool:
    # Enable if env var is set or if logger is enabled
    return os.getenv(LOG_ENV_VAR, "0") == "1" or logger.isEnabledFor(logging.INFO)

class ConfigAccessLogger:
    """Proxy/wrapper that logs attribute access for config objects."""
    def __init__(self, obj, prefix="config"):
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_prefix", prefix)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._obj, name)
        if is_logging_enabled():
            logger.info(f"{self._prefix}.{name} -> {repr(value)}")
        # Recursively wrap dataclasses or dicts
        if hasattr(value, "__dataclass_fields__"):
            return ConfigAccessLogger(value, f"{self._prefix}.{name}")
        return value

    def __getitem__(self, key):
        value = self._obj[key]
        if is_logging_enabled():
            logger.info(f"{self._prefix}[{repr(key)}] -> {repr(value)}")
        return value

    def __setattr__(self, name, value):
        setattr(self._obj, name, value)
        if is_logging_enabled():
            logger.info(f"{self._prefix}.{name} SET TO {repr(value)}")

    def __repr__(self):
        return f"<ConfigAccessLogger {self._prefix}: {repr(self._obj)}>"
