"""Moderator prefab for debates."""

from collections.abc import Mapping
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.associative_memory import formative_memories
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib
import numpy as np

DEFAULT_GOAL_COMPONENT_KEY = "goal"
DEFAULT_CONTEXT_COMPONENT_KEY = "context"
DEFAULT_RULES_COMPONENT_KEY = "rules"


@dataclasses.dataclass
class ModeratorEntity(prefab_lib.Prefab):
    """Prefab for a neutral moderator."""

    description: str = (
        "An entity representing a neutral debate moderator."
    )
    params: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: {
            "name": "Moderator",
            "goal": "Ensure fair and orderly debate.",
            "context": "Experienced moderator.",
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the moderator agent."""
        agent_name = self.params.get("name")
        goal = self.params.get("goal")
        context = self.params.get("context")

        observation_to_memory = agent_components.observation.ObservationToMemory()
        observation = agent_components.observation.LastNObservations(
            history_length=20,
            pre_act_label="\nDebate log:"
        )

        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            def embed_fn(text: str) -> np.ndarray:
                return embedder.encode(text)
        except Exception:
            def embed_fn(text: str) -> np.ndarray:  # pragma: no cover - fallback
                return np.random.randn(384)

        memory_factory = formative_memories.FormativeMemoryFactory(
            model=model,
            embedder=embed_fn,
            shared_memories=[
                f"Name: {agent_name}",
                f"Role: {goal}",
                f"Background: {context}",
            ],
        )
        raw_memory_bank = memory_factory._blank_memory_factory_call()

        for mem in [
            f"I am {agent_name}, a neutral moderator.",
            f"Role: {goal}",
            f"Background: {context}",
            "Assess arguments on clarity, evidence and logic.",
        ]:
            raw_memory_bank.add(mem)

        memory = agent_components.memory.AssociativeMemory(raw_memory_bank)

        goal_component = agent_components.constant.Constant(
            state=goal,
            pre_act_label=f"\n{agent_name}'s role:"
        )
        context_component = agent_components.constant.Constant(
            state=context,
            pre_act_label=f"\nBackground for {agent_name}:"
        )
        rules_component = agent_components.constant.Constant(
            state="Maintain neutrality and keep debate on track.",
            pre_act_label="\nModeration rules:"
        )

        components = {
            "observation_to_memory": observation_to_memory,
            agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY: observation,
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: memory,
            DEFAULT_GOAL_COMPONENT_KEY: goal_component,
            DEFAULT_CONTEXT_COMPONENT_KEY: context_component,
            DEFAULT_RULES_COMPONENT_KEY: rules_component,
        }

        component_order = [
            DEFAULT_CONTEXT_COMPONENT_KEY,
            DEFAULT_GOAL_COMPONENT_KEY,
            DEFAULT_RULES_COMPONENT_KEY,
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
            agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY,
        ]

        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=component_order,
        )

        return entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=agent_name,
            act_component=act_component,
            context_components=components,
        )
