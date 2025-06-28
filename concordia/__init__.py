"""Concordia - A framework for creating multi-agent simulations and environments.

This library provides tools for building complex agent-based simulations with:
- Agents with memories, personalities, and decision-making capabilities
- Associative memory systems for long-term knowledge storage
- Clock|timing mechanisms for simulation control
- Document management for simulation inputs and outputs
- Environment constructs for agent interactions
- Language model integrations
- Thought chain processing for agent reasoning
"""

__version__ = "1.0.0"

# Import commonly used modules to make them available directly from concordia
from concordia.agents import entity_agent
from concordia.associative_memory import basic_associative_memory, formative_memories
from concordia.clocks import game_clock
from concordia.document import document, interactive_document
from concordia.environment import engine
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains


