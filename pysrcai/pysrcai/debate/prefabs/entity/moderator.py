"""Generic debate moderator prefab."""

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
class Moderator(prefab_lib.Prefab):
    """A neutral debate moderator."""

    description: str = (
        "An entity responsible for moderating a debate and judging the winner."
    )
    params: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: {
            "name": "Moderator",
            "goal": "Ensure fair and orderly discussion.",
            "context": "",
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the moderator agent."""

        name = self.params.get("name")
        goal = self.params.get("goal")
        context = self.params.get("context", "")

        embedder = SentenceTransformer("all-MiniLM-L6-v2")

        def embed_fn(text: str) -> np.ndarray:
            return embedder.encode(text)

        memory_factory = formative_memories.FormativeMemoryFactory(
            model=model,
            embedder=embed_fn,
            shared_memories=[f"Name: {name}", f"Role: {goal}", f"Context: {context}"],
        )
        raw_memory_bank = memory_factory._blank_memory_factory_call()
        memory = agent_components.memory.AssociativeMemory(raw_memory_bank)

        observation = agent_components.observation.LastNObservations(
            history_length=20, pre_act_label="\nRecent discussion:"
        )
        observation_to_memory = agent_components.observation.ObservationToMemory()

        judgment_criteria = (
            "Assessment criteria:\n"
            "1. Logical consistency\n"
            "2. Factual evidence\n"
            "3. Persuasiveness\n"
        )

        components = {
            "observation_to_memory": observation_to_memory,
            agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY: observation,
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: memory,
            "goal": agent_components.constant.Constant(
                state=goal, pre_act_label="\nRole:"
            ),
            "context": agent_components.constant.Constant(
                state=context, pre_act_label="\nBackground:"
            ),
            "judgment": agent_components.constant.Constant(
                state=judgment_criteria, pre_act_label="\nJudgment criteria:"
            ),
        }

        component_order = [
            "context",
            "goal",
            "judgment",
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
