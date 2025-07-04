"""
PySrcAI: A Python-based AI agent framework for simulations, embeddings, and more.
"""

__version__ = "0.1.0"

# Core components
from . import agents
from . import config
from . import core
from . import embeddings
from . import llm
from . import utils

# Main public API
from .core.factory import SimulationFactory
from .core.engine import SimulationEngine, SequentialEngine
from .agents.base import Agent, Actor, Archon
from .llm.language_model import LanguageModel

__all__ = [
    # Core classes
    "SimulationFactory",
    "SimulationEngine", 
    "SequentialEngine",
    "Agent",
    "Actor",
    "Archon",
    "LanguageModel",
    
    # Modules
    "agents",
    "config", 
    "core",
    "embeddings",
    "llm",
    "utils",
]
