"""PySrcAI Agents Module.

This module provides the core agent hierarchy for PySrcAI simulations:

- Agent: Base abstract class for all autonomous entities
- Actor: Specialized agents that participate in simulations
- Archon: Specialized agents that moderate and orchestrate simulations

The agent hierarchy is designed to provide clear separation of concerns:
- Actors focus on participation, goals, and competitive/collaborative behavior
- Archons focus on moderation, rule enforcement, and simulation management

Usage Example:
    from pysrcai.agents.base import Actor, Archon
    
    # Create a debate participant
    participant = Actor(
        agent_name="Alice",
        goals=["Argue for renewable energy"],
        personality_traits={"assertiveness": 0.8, "knowledge_level": 0.9}
    )
    
    # Create a debate moderator
    moderator = Archon(
        agent_name="DebateMod",
        moderation_rules=["Enforce 2-minute time limits", "Maintain civility"],
        authority_level="high"
    )
"""

from .base.agent import (
    # Core classes
    Agent,
    AgentWithLogging,
    
    # Component classes
    BaseComponent,
    ContextComponent,
    ActingComponent,
    
    # Action specification classes
    ActionSpec,
    OutputType,
    Phase,
    
    # Type definitions
    ComponentName,
    ComponentContext,
    ComponentContextMapping,
    ComponentState,
    
    # Action type groups
    ACTOR_ACTION_TYPES,
    ARCHON_ACTION_TYPES,
    FREE_ACTION_TYPES,
    CHOICE_ACTION_TYPES,
    
    # Convenience functions
    free_action_spec,
    choice_action_spec,
    float_action_spec,
    speech_action_spec,
)

from .base.actor import (
    Actor,
    ActorWithLogging,
    ActorActingComponent,
)

from .base.archon import (
    Archon,
    ArchonWithLogging,
    ArchonActingComponent,
)

from ..llm.llm_components import (
    LLMActingComponent,
    ActorLLMComponent,
    ArchonLLMComponent,
    ConfigurableLLMComponent,
)

from .memory import (
    MemoryBank,
    BasicMemoryBank,
    AssociativeMemoryBank,
    MemoryComponent,
)

__all__ = [
    # Core agent classes
    "Agent",
    "AgentWithLogging",
    "Actor",
    "ActorWithLogging", 
    "Archon",
    "ArchonWithLogging",
    
    # Component classes
    "BaseComponent",
    "ContextComponent",
    "ActingComponent",
    "ActorActingComponent",
    "ArchonActingComponent",
    
    # LLM-powered components
    "LLMActingComponent",
    "ActorLLMComponent",
    "ArchonLLMComponent",
    "ConfigurableLLMComponent",
    
    # Memory components
    "MemoryBank",
    "BasicMemoryBank",
    "AssociativeMemoryBank", 
    "MemoryComponent",
    
    # Action specification
    "ActionSpec",
    "OutputType",
    "Phase",
    
    # Type definitions
    "ComponentName",
    "ComponentContext", 
    "ComponentContextMapping",
    "ComponentState",
    
    # Action type groups
    "ACTOR_ACTION_TYPES",
    "ARCHON_ACTION_TYPES",
    "FREE_ACTION_TYPES",
    "CHOICE_ACTION_TYPES",
    
    # Default specs
    "DEFAULT_ACTION_SPEC",
    "DEFAULT_SPEECH_ACTION_SPEC",
    
    # Convenience functions
    "free_action_spec",
    "choice_action_spec", 
    "float_action_spec",
    "speech_action_spec",
]
