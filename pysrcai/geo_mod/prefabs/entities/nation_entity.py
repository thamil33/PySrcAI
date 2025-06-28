"""Prefab for a nation-state entity."""

from collections.abc import Mapping
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.associative_memory import formative_memories
from concordia.components import agent as agent_components
import numpy as np
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib
from sentence_transformers import SentenceTransformer

DEFAULT_GOAL_COMPONENT_KEY = 'goal'
DEFAULT_CONTEXT_COMPONENT_KEY = 'context'


@dataclasses.dataclass
class NationEntity(prefab_lib.Prefab):
    """A prefab for a nation-state entity.

    This prefab is designed to be highly configurable to represent different
    nations with unique goals, histories, and political contexts.
    """

    description: str = (
        'An entity representing a nation-state in a geopolitical simulation. '
        'It is defined by its name, goals, and a summary of its current '
        'political and historical context.'
    )
    params: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: {
            'name': 'Default Nation',
            'goal': 'Maintain peace and prosperity.',
            'context': 'This nation has a long history of diplomacy.',
            'word_limits': None,
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the nation agent.

        Args:
            model: The language model to use for generating responses.
            memory_bank: The agent's memory bank.

        Returns:
            An entity agent representing the nation.
        """
        agent_name = self.params.get('name')
        goal = self.params.get('goal')
        context = self.params.get('context')
        word_limits = self.params.get('word_limits')

        # 1. Core Components for Memory and Observation
        observation_to_memory = agent_components.observation.ObservationToMemory()
        observation = agent_components.observation.LastNObservations(
            history_length=10,
            pre_act_label=f'\nRecent events and statements:'
        )

        # Create a memory bank with formative memories for the nation
        # First, set up embedder function for the memory
        embedder = SentenceTransformer("all-MiniLM-L6-v2")  # or your preferred model

        def simple_embedder(text: str) -> np.ndarray:
            return embedder.encode(text)

        # Create memory bank using FormativeMemoryFactory
        memory_factory = formative_memories.FormativeMemoryFactory(
            model=model,
            embedder=simple_embedder,
            shared_memories=[
                f"Name: {agent_name}",
                f"Goal: {goal}",
                f"Context: {context}",
            ],
        )

        # Create the raw memory bank
        raw_memory_bank = memory_factory._blank_memory_factory_call()

        # Generate appropriate response limit text based on word_limits
        if word_limits:
            opening = word_limits['opening_statements']
            rebuttals = word_limits['rebuttals']
            final = word_limits['final_arguments']
            
            response_limit = (f"RESPONSE LIMIT: I will limit my opening statements to {opening['min']}-{opening['max']} words, "
                            f"rebuttals to {rebuttals['min']}-{rebuttals['max']} words, and final arguments to "
                            f"{final['min']}-{final['max']} words to maintain clarity and respect time constraints.")
        else:
            # Fallback for when no word limits are provided
            response_limit = "RESPONSE LIMIT: I will limit my statements to 200-300 words to maintain clarity and respect time."

        # Add the nation's core memories to the raw bank
        for mem in [
            f"I am {agent_name}, representing my nation's interests.",
            f"My primary goal: {goal}",
            f"My context and background: {context}",
            "IMPORTANT: When speaking in debates, I must keep my responses concise and focused.",
            response_limit,
            "I must make every word count and present only the strongest arguments.",
        ]:
            raw_memory_bank.add(mem)

        # Wrap the memory bank in an AssociativeMemory component
        memory = agent_components.memory.AssociativeMemory(raw_memory_bank)

        # 2. Nation-Specific Components (from params)
        goal_component = agent_components.constant.Constant(
            state=goal,
            pre_act_label=f'\n{agent_name}\'s primary goal:'
        )
        context_component = agent_components.constant.Constant(
            state=context,
            pre_act_label=f'\nKey context for {agent_name}:'
        )

        # 3. Assemble the Agent's "Mind"
        components_of_agent = {
            'observation_to_memory': observation_to_memory,
            agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY: observation,
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: memory,
            DEFAULT_GOAL_COMPONENT_KEY: goal_component,
            DEFAULT_CONTEXT_COMPONENT_KEY: context_component,
        }

        # The order determines how the final prompt is constructed.
        component_order = [
            DEFAULT_CONTEXT_COMPONENT_KEY,
            DEFAULT_GOAL_COMPONENT_KEY,
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
            agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY,
        ]

        # 4. Create the Act Component
        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=component_order,
        )

        # 5. Build and return the final agent object
        agent = entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=agent_name,
            act_component=act_component,
            context_components=components_of_agent,
        )

        return agent
