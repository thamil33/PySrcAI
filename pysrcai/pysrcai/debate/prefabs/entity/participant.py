"""Generic debate participant prefab."""

from collections.abc import Mapping
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.associative_memory import formative_memories
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclasses.dataclass
class Participant(prefab_lib.Prefab):
    """A generic debate participant."""

    description: str = (
        "An entity representing a debate participant with a goal and context."
    )
    params: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: {
            "name": "Participant",
            "goal": "",
            "context": "",
            "word_limits": None,
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the participant agent."""

        name = self.params.get("name")
        goal = self.params.get("goal", "")
        context = self.params.get("context", "")
        word_limits = self.params.get("word_limits")

        embedder = SentenceTransformer("all-MiniLM-L6-v2")

        def embed_fn(text: str) -> np.ndarray:
            return embedder.encode(text)

        memory_factory = formative_memories.FormativeMemoryFactory(
            model=model,
            embedder=embed_fn,
            shared_memories=[f"Name: {name}", f"Goal: {goal}", f"Context: {context}"],
        )
        raw_memory_bank = memory_factory._blank_memory_factory_call()

        if word_limits:
            opening = word_limits.get("opening", {"min": 0, "max": 50})
            raw_memory_bank.add(
                f"Responses should be {opening['min']}-{opening['max']} words."
            )

        memory = agent_components.memory.AssociativeMemory(raw_memory_bank)

        observation = agent_components.observation.LastNObservations(
            history_length=10, pre_act_label="\nRecent discussion:"
        )
        observation_to_memory = agent_components.observation.ObservationToMemory()

        components = {
            "observation_to_memory": observation_to_memory,
            agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY: observation,
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: memory,
            "goal": agent_components.constant.Constant(
                state=goal, pre_act_label="\nGoal:"
            ),
            "context": agent_components.constant.Constant(
                state=context, pre_act_label="\nContext:"
            ),
        }

        component_order = [
            "context",
            "goal",
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
            agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY,
        ]

        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=component_order,
        )

        return entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=act_component,
            context_components=components,
        )
