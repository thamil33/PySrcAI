"""LLM-powered acting components for PySrcAI agents.

This module provides acting components that use language models to make
intelligent decisions for agents. It includes specialized components for
Actor and Archon roles with appropriate prompt engineering.
"""

import abc
from typing import Any
from collections.abc import Mapping

from pysrcai.src.agents import ActingComponent, ActionSpec, ComponentContextMapping, OutputType

from pysrcai.src.language_model_client import (
    LMStudioLanguageModel,
    OpenRouterLanguageModel,
    NoLanguageModel,
)
from pysrcai.src.language_model_client.language_model import LanguageModel


class LLMActingComponent(ActingComponent):
    """Base class for LLM-powered acting components.
    
    This component uses a language model to make action decisions based on
    context from other components and the action specification.
    """
    
    def __init__(
        self,
        language_model: LanguageModel,
        *,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ):
        """Initialize the LLM acting component.
        
        Args:
            language_model: The language model to use for decision making.
            temperature: Temperature for text generation (0.0 to 1.0).
            max_tokens: Maximum tokens in the response.
        """
        super().__init__()
        self._language_model = language_model
        self._temperature = temperature
        self._max_tokens = max_tokens
    
    @abc.abstractmethod
    def _build_prompt(
        self,
        context: ComponentContextMapping,
        action_spec: ActionSpec,
    ) -> str:
        """Build the prompt for the language model.
        
        Args:
            context: Context from all the agent's components.
            action_spec: The specification for the action.
            
        Returns:
            The prompt string to send to the language model.
        """
        raise NotImplementedError()
    
    def _format_context(self, context: ComponentContextMapping) -> str:
        """Format context from components into a readable string.
        
        Args:
            context: Context from all the agent's components.
            
        Returns:
            Formatted context string.
        """
        if not context:
            return "No additional context available."
        
        formatted_parts = []
        for component_name, component_context in context.items():
            if component_context.strip():  # Only include non-empty context
                formatted_parts.append(f"{component_name}: {component_context}")
        
        return "\n".join(formatted_parts) if formatted_parts else "No additional context available."
    
    def get_action_attempt(
        self,
        context: ComponentContextMapping,
        action_spec: ActionSpec,
    ) -> str:
        """Use the LLM to decide the action.
        
        Args:
            context: Context from all the agent's components.
            action_spec: The specification for the action.
            
        Returns:
            The action that the agent should attempt.
        """
        # Build the prompt using the subclass implementation
        prompt = self._build_prompt(context, action_spec)
        
        # Handle choice-based actions differently
        if action_spec.output_type in (OutputType.CHOICE, OutputType.DECISION, 
                                      OutputType.EVALUATE, OutputType.TERMINATE):
            if action_spec.options:
                # Use choice sampling for multiple choice questions
                choice_idx, choice, _ = self._language_model.sample_choice(
                    prompt, 
                    list(action_spec.options)
                )
                return choice
        
        # For free-form actions, use text sampling
        response = self._language_model.sample_text(
            prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        
        return response.strip()


class ActorLLMComponent(LLMActingComponent):
    """LLM acting component specialized for Actor agents.
    
    This component crafts prompts that emphasize the Actor's role as a
    simulation participant with goals, personality, and competitive/collaborative
    objectives.
    """
    
    def _build_prompt(
        self,
        context: ComponentContextMapping,
        action_spec: ActionSpec,
    ) -> str:
        """Build a prompt tailored for Actor agents.
        
        Args:
            context: Context from all the agent's components.
            action_spec: The specification for the action.
            
        Returns:
            The prompt string optimized for Actor decision-making.
        """

        from .actor import Actor 
        


        agent = self.get_agent()
        agent_name = agent.name
        
        # Get Actor-specific context if available
        actor_context = {}
        if isinstance(agent, Actor):
            actor_context = agent.get_actor_context()
        
        # Format the prompt with Actor-specific framing
        prompt_parts = [
            f"You are {agent_name}, a participant in a simulation.",
        ]
        
        # Add goals if available
        if 'goals' in actor_context and actor_context['goals']:
            goals_str = ', '.join(actor_context['goals'])
            prompt_parts.append(f"Your goals are: {goals_str}")
        
        # Add personality traits if available
        if 'personality_traits' in actor_context and actor_context['personality_traits']:
            traits = actor_context['personality_traits']
            traits_str = ', '.join(f"{k}: {v}" for k, v in traits.items())
            prompt_parts.append(f"Your personality traits: {traits_str}")
        
        # Add component context
        formatted_context = self._format_context(context)
        if formatted_context != "No additional context available.":
            prompt_parts.append(f"Current context:\n{formatted_context}")
        
        # Add the action specification
        call_to_action = action_spec.call_to_action.format(name=agent_name)
        prompt_parts.append(f"\n{call_to_action}")
        
        # Add action-specific guidance
        if action_spec.output_type == OutputType.SPEECH:
            prompt_parts.append(
                f"Respond as {agent_name} would speak, staying true to your goals and personality. "
                f"Be direct and engaging."
            )
        elif action_spec.output_type in (OutputType.ACTION, OutputType.FREE):
            prompt_parts.append(
                f"Decide what {agent_name} would do next, considering your goals and the situation. "
                f"Be specific and strategic."
            )
        elif action_spec.output_type == OutputType.DECISION:
            prompt_parts.append(
                f"Make a decision as {agent_name}, weighing the options against your goals."
            )
        
        return "\n\n".join(prompt_parts)


class ArchonLLMComponent(LLMActingComponent):
    """LLM acting component specialized for Archon agents.
    
    This component crafts prompts that emphasize the Archon's role as a
    simulation moderator with authority, rules, and orchestration responsibilities.
    """
    
    def _build_prompt(
        self,
        context: ComponentContextMapping,
        action_spec: ActionSpec,
    ) -> str:
        """Build a prompt tailored for Archon agents.
        
        Args:
            context: Context from all the agent's components.
            action_spec: The specification for the action.
            
        Returns:
            The prompt string optimized for Archon moderation and orchestration.
        """
        from .archon import Archon  

        agent = self.get_agent()
        agent_name = agent.name
        
        # Get Archon-specific context if available
        archon_context = {}
        if isinstance(agent, Archon):
            archon_context = agent.get_archon_context()
        
        # Format the prompt with Archon-specific framing
        prompt_parts = [
            f"You are {agent_name}, a moderator/orchestrator in a simulation.",
            "Your role is to maintain fair play, enforce rules, and manage interactions."
        ]
        
        # Add authority level if available
        if 'authority_level' in archon_context:
            level = archon_context['authority_level']
            prompt_parts.append(f"Your authority level: {level}")
        
        # Add moderation rules if available
        if 'moderation_rules' in archon_context and archon_context['moderation_rules']:
            rules_str = '\n- '.join(archon_context['moderation_rules'])
            prompt_parts.append(f"Your moderation rules:\n- {rules_str}")
        
        # Add managed entities if available
        if 'managed_entities' in archon_context and archon_context['managed_entities']:
            entities_str = ', '.join(archon_context['managed_entities'])
            prompt_parts.append(f"You are managing: {entities_str}")
        
        # Add session state if available
        if 'session_state' in archon_context:
            state = archon_context['session_state']
            prompt_parts.append(f"Current session state: {state}")
        
        # Add component context
        formatted_context = self._format_context(context)
        if formatted_context != "No additional context available.":
            prompt_parts.append(f"Current context:\n{formatted_context}")
        
        # Add the action specification
        call_to_action = action_spec.call_to_action.format(name=agent_name)
        prompt_parts.append(f"\n{call_to_action}")
        
        # Add action-specific guidance
        if action_spec.output_type == OutputType.MODERATE:
            prompt_parts.append(
                f"Provide a moderation decision that maintains fairness and follows your rules. "
                f"Be clear and authoritative but fair."
            )
        elif action_spec.output_type == OutputType.EVALUATE:
            prompt_parts.append(
                f"Evaluate the situation objectively based on your criteria and rules. "
                f"Provide a fair assessment."
            )
        elif action_spec.output_type == OutputType.ORCHESTRATE:
            prompt_parts.append(
                f"Orchestrate the next phase of the simulation. Consider all participants "
                f"and maintain engagement while following protocols."
            )
        elif action_spec.output_type == OutputType.TERMINATE:
            prompt_parts.append(
                f"Decide whether to continue or terminate the session. Consider completion "
                f"criteria, participant engagement, and overall objectives."
            )
        
        return "\n\n".join(prompt_parts)


class ConfigurableLLMComponent(LLMActingComponent):
    """Configurable LLM acting component for custom agent types.
    
    This component allows for custom prompt templates and is useful for
    specialized agent types that don't fit the Actor/Archon distinction.
    """
    
    def __init__(
        self,
        language_model: LanguageModel,
        *,
        role_description: str,
        prompt_template: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ):
        """Initialize the configurable LLM acting component.
        
        Args:
            language_model: The language model to use for decision making.
            role_description: Description of the agent's role.
            prompt_template: Optional custom prompt template. Should include
                {name}, {role}, {context}, and {call_to_action} placeholders.
            temperature: Temperature for text generation (0.0 to 1.0).
            max_tokens: Maximum tokens in the response.
        """
        super().__init__(language_model, temperature=temperature, max_tokens=max_tokens)
        self._role_description = role_description
        self._prompt_template = prompt_template or self._default_template()
    
    def _default_template(self) -> str:
        """Default prompt template."""
        return (
            "You are {name}, {role}\n\n"
            "Current context:\n{context}\n\n"
            "{call_to_action}\n\n"
            "Respond appropriately to your role and the situation."
        )
    
    def _build_prompt(
        self,
        context: ComponentContextMapping,
        action_spec: ActionSpec,
    ) -> str:
        """Build a prompt using the configured template.
        
        Args:
            context: Context from all the agent's components.
            action_spec: The specification for the action.
            
        Returns:
            The prompt string based on the template.
        """
        agent = self.get_agent()
        agent_name = agent.name
        formatted_context = self._format_context(context)
        call_to_action = action_spec.call_to_action.format(name=agent_name)
        
        return self._prompt_template.format(
            name=agent_name,
            role=self._role_description,
            context=formatted_context,
            call_to_action=call_to_action,
        )
def create_language_model(model_type: str = "mock"):
    if model_type == "lmstudio":
        return LMStudioLanguageModel(
            model_name="local-model",
            base_url="http://localhost:1234/v1",
            verbose_logging=True
        )
    elif model_type == "openrouter":
        return OpenRouterLanguageModel(
            model_name="mistralai/mistral-7b-instruct:free",
            verbose_logging=True
        )
    else:
        return NoLanguageModel()

