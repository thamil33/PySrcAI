"""Prefab for a debate moderator game master."""

from collections.abc import Sequence
import dataclasses
from typing import Any, Dict, List, Mapping, Optional

from concordia.associative_memory import basic_associative_memory
from concordia.components.agent import memory as memory_component
from concordia.components.agent import observation as observation_component
from concordia.components.agent import constant as constant_component
from concordia.components.agent import concat_act_component
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib
from concordia.agents import entity_agent_with_logging


@dataclasses.dataclass
class DebateSettings:
    """Settings for the debate."""
    max_turns: int = 4
    acting_order: str = "fixed"  # can be "fixed" or "random"


@dataclasses.dataclass
class ModeratorGmPrefab(prefab_lib.Prefab):
    """Prefab for a moderator game master."""

    description: str = "A game master that moderates a turn-based debate."
    params: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "name": "Moderator",
            "max_turns": 4,
            "acting_order": "fixed",
            "verbose": True,
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_lib.Entity:
        """Build the debate moderator game master.

        Args:
            model: Language model for the game master
            memory_bank: Memory bank for the game master

        Returns:
            The configured game master entity
        """
        name = self.params.get("name", "Moderator")
        max_turns = self.params.get("max_turns", 4)
        acting_order = self.params.get("acting_order", "fixed")
        verbose = self.params.get("verbose", True)

        # Use the entities passed to the prefab
        entities = getattr(self, "entities", [])

        # Create debate settings
        settings = DebateSettings(
            max_turns=max_turns,
            acting_order=acting_order
        )

        # Create components
        components = {}

        # Add memory component
        components["__memory__"] = memory_component.AssociativeMemory(memory_bank)

        # Add observation component that will broadcast observations to entities
        components[observation_component.DEFAULT_OBSERVATION_COMPONENT_KEY] = observation_component.ObservationToMemory()

        # Add system prompt as a constant component
        system_prompt = (
            f"You are a debate moderator named {name}. "
            "Your role is to facilitate a structured debate between multiple parties. "
            f"This debate will consist of {max_turns} turns. "
            "Ensure each participant gets an equal opportunity to speak, and maintain a respectful tone."
        )

        components["system_prompt"] = constant_component.Constant(
            state=system_prompt,
            pre_act_label="\nSystem prompt:"
        )

        # Set up components order for the action component
        component_order = [
            "system_prompt",
            memory_component.DEFAULT_MEMORY_COMPONENT_KEY,
            observation_component.DEFAULT_OBSERVATION_COMPONENT_KEY,
        ]

        # Create the act component
        act_component = concat_act_component.ConcatActComponent(
            model=model,
            component_order=component_order,
        )

        # Create the game master entity
        gm = entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components,
        )

        return gm