"""Actor class for PySrcAI - simulation participants.

Actors are specialized agents that represent direct participants in simulations.
They are goal-oriented entities that compete, collaborate, negotiate, or engage
in other interactive behaviors within the simulation environment.

Examples of Actors:
- Debate participants arguing positions
- Players in strategy games  
- Negotiators in business scenarios
- Characters in social simulations
- Competitors in tournaments

Actors are distinguished from Archons (moderators) by their participatory role
rather than administrative function.
"""

from typing import Any
from collections.abc import Mapping

from .agent import Agent, ActingComponent, ContextComponent, ActionSpec, OutputType
from ...llm.llm_components import ActorLLMComponent


class ActorActingComponent(ActingComponent):
    """Default acting component for Actor agents.
    
    This provides basic action decision-making for Actor agents. It can be
    replaced with more sophisticated acting components for specific use cases.
    """
    
    def get_action_attempt(
        self,
        context: Mapping[str, str],
        action_spec: ActionSpec,
    ) -> str:
        """Makes action decisions for Actor agents.
        
        Args:
            context: Context from all the agent's components.
            action_spec: The specification for the action.
            
        Returns:
            The action that the actor should attempt.
        """
        # Basic implementation - combine all context and provide to action decision
        combined_context = "\n".join(f"{name}: {ctx}" for name, ctx in context.items())
        
        # This is a placeholder - in practice, this would use the language model
        # to make intelligent decisions based on the context and action spec
        if action_spec.output_type == OutputType.CHOICE and action_spec.options:
            # For choice actions, return the first option (placeholder logic)
            return action_spec.options[0]
        elif action_spec.output_type == OutputType.FLOAT:
            return "0.5"  # Placeholder
        else:
            # For free-form actions, return a placeholder
            agent_name = self.get_agent().name
            return f"{agent_name} considers the situation carefully."


class Actor(Agent):
    """A specialized agent representing simulation participants.
    
    Actors are the primary participants in PySrcAI simulations. They are
    goal-oriented entities that engage with other actors and respond to
    the simulation environment. Unlike Archons, Actors are not responsible
    for managing the simulation itself.
    
    Key characteristics of Actors:
    - Participatory role in simulations
    - Goal-oriented behavior
    - Can compete, collaborate, or negotiate
    - Respond to environmental changes
    - Have personal motivations and objectives
    
    Actors can be configured with different:
    - Acting strategies (via ActingComponent)
    - Memory systems (via ContextComponents)
    - Behavioral patterns (via configuration)
    - Personality traits and goals
    """
    
    def __init__(
        self,
        agent_name: str,
        act_component: ActingComponent | None = None,
        context_components: Mapping[str, ContextComponent] | None = None,
        goals: list[str] | None = None,
        personality_traits: dict[str, Any] | None = None,
        language_model: Any = None,  # LanguageModel type, but avoiding circular import
    ):
        """Initializes an Actor agent.
        
        Args:
            agent_name: The name of the actor.
            act_component: The component responsible for action decisions.
                If None and language_model is provided, uses ActorLLMComponent.
                If both are None, uses default ActorActingComponent.
            context_components: Optional context components for the actor.
            goals: Optional list of goals for this actor.
            personality_traits: Optional personality traits affecting behavior.
            language_model: Optional language model for LLM-powered decision making.
                If provided and act_component is None, creates an ActorLLMComponent.
        """
        # Create appropriate acting component
        if act_component is None:
            if language_model is not None:
                act_component = ActorLLMComponent(language_model)
            else:
                act_component = ActorActingComponent()
            
        super().__init__(
            agent_name=agent_name,
            act_component=act_component,
            context_components=context_components,
        )
        
        # Actor-specific attributes
        self._goals = list(goals or [])
        self._personality_traits = dict(personality_traits or {})
        self._role = "participant"  # Distinguishes from Archon's "moderator" role
    
    @property
    def goals(self) -> list[str]:
        """Returns the actor's current goals."""
        return self._goals.copy()
    
    def add_goal(self, goal: str) -> None:
        """Adds a new goal for the actor.
        
        Args:
            goal: The goal to add.
        """
        if goal not in self._goals:
            self._goals.append(goal)
    
    def remove_goal(self, goal: str) -> None:
        """Removes a goal from the actor.
        
        Args:
            goal: The goal to remove.
        """
        if goal in self._goals:
            self._goals.remove(goal)
    
    @property
    def personality_traits(self) -> dict[str, Any]:
        """Returns the actor's personality traits."""
        return self._personality_traits.copy()
    
    def set_personality_trait(self, trait: str, value: Any) -> None:
        """Sets a personality trait for the actor.
        
        Args:
            trait: The trait name.
            value: The trait value.
        """
        self._personality_traits[trait] = value
    
    @property
    def role(self) -> str:
        """Returns the agent's role type (always 'participant' for Actors)."""
        return self._role
    
    def get_actor_context(self) -> dict[str, Any]:
        """Returns Actor-specific context information.
        
        This method provides Actor-specific information that can be used
        by components or other systems that need to understand this agent's
        participatory role and characteristics.
        
        Returns:
            Dictionary containing actor-specific context.
        """
        return {
            "role": self._role,
            "goals": self._goals,
            "personality_traits": self._personality_traits,
            "agent_type": "Actor",
        }
        
class ActorWithLogging(Actor):
    """An Actor that includes logging capabilities for debugging and monitoring."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._action_log: list[dict[str, Any]] = []
        self._observation_log: list[str] = []
    
    def act(self, action_spec: ActionSpec) -> str:
        """Acts and logs the action for debugging."""
        action = super().act(action_spec)
        
        # Log the action
        log_entry = {
            "action": action,
            "action_spec": {
                "call_to_action": action_spec.call_to_action,
                "output_type": action_spec.output_type.name,
                "options": action_spec.options,
                "tag": action_spec.tag,
            },
            "phase": self.get_phase().name,
            "goals": self._goals,
        }
        self._action_log.append(log_entry)
        
        return action
    
    def observe(self, observation: str) -> None:
        """Observes and logs the observation for debugging."""
        super().observe(observation)
        self._observation_log.append(observation)
    
    def get_last_log(self) -> dict[str, Any]:
        """Returns the most recent debugging information."""
        return {
            "agent_name": self.name,
            "agent_type": "Actor",
            "role": self._role,
            "goals": self._goals,
            "personality_traits": self._personality_traits,
            "recent_actions": self._action_log[-5:],  # Last 5 actions
            "recent_observations": self._observation_log[-5:],  # Last 5 observations
            "current_phase": self.get_phase().name,
            "component_count": len(self.get_all_context_components()),
        }
    
    def get_full_log(self) -> dict[str, Any]:
        """Returns complete logging information."""
        return {
            "agent_name": self.name,
            "agent_type": "Actor", 
            "role": self._role,
            "goals": self._goals,
            "personality_traits": self._personality_traits,
            "all_actions": self._action_log,
            "all_observations": self._observation_log,
            "current_phase": self.get_phase().name,
            "components": list(self.get_all_context_components().keys()),
        }
