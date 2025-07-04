"""Archon class for PySrcAI - simulation moderators and orchestrators.

Archons are specialized agents that manage and moderate simulations rather than
participating directly. They handle administrative functions, enforce rules,
orchestrate interactions, and manage the simulation environment.

Examples of Archons:
- Debate moderators managing discussion flow
- Game masters controlling game state
- Environment controllers managing scenarios
- Judges evaluating performances
- Session orchestrators coordinating multi-agent interactions

Archons are distinguished from Actors (participants) by their administrative
and oversight functions rather than competitive/collaborative participation.
"""

from typing import Any
from collections.abc import Mapping, Sequence

from .agent import Agent, ActingComponent, ContextComponent, ActionSpec, OutputType
from .llm_components import ArchonLLMComponent


class ArchonActingComponent(ActingComponent):
    """Default acting component for Archon agents.
    
    This provides basic moderation and orchestration decision-making for Archon
    agents. It can be replaced with more sophisticated acting components for
    specific moderation use cases.
    """
    
    def get_action_attempt(
        self,
        context: Mapping[str, str],
        action_spec: ActionSpec,
    ) -> str:
        """Makes moderation/orchestration decisions for Archon agents.
        
        Args:
            context: Context from all the agent's components.
            action_spec: The specification for the action.
            
        Returns:
            The action that the archon should attempt.
        """
        # Basic implementation - combine all context for decision making
        combined_context = "\n".join(f"{name}: {ctx}" for name, ctx in context.items())
        
        # This is a placeholder - in practice, this would use the language model
        # to make intelligent moderation/orchestration decisions
        if action_spec.output_type == OutputType.CHOICE and action_spec.options:
            # For choice actions, return the first option (placeholder logic)
            return action_spec.options[0]
        elif action_spec.output_type == OutputType.FLOAT:
            return "0.5"  # Placeholder
        elif action_spec.output_type == OutputType.TERMINATE:
            return "continue"  # Default to continuing unless there's a reason to stop
        elif action_spec.output_type in (OutputType.MODERATE, OutputType.ORCHESTRATE):
            agent_name = self.get_agent().name
            return f"{agent_name} observes the situation and maintains order."
        else:
            # For other actions, return a placeholder
            agent_name = self.get_agent().name
            return f"{agent_name} monitors the situation."


class Archon(Agent):
    """A specialized agent representing simulation moderators and orchestrators.
    
    Archons are responsible for managing simulations rather than participating
    in them directly. They enforce rules, moderate interactions, orchestrate
    complex scenarios, and maintain the simulation environment.
    
    Key characteristics of Archons:
    - Administrative/oversight role in simulations
    - Rule enforcement and fair play
    - Orchestration of multi-agent interactions
    - Environment and scenario management
    - Evaluation and judgment capabilities
    
    Archons can be configured with different:
    - Moderation strategies (via ActingComponent)
    - Rule systems (via ContextComponents)
    - Evaluation metrics (via configuration)
    - Orchestration patterns and templates
    """
    
    def __init__(
        self,
        agent_name: str,
        act_component: ActingComponent | None = None,
        context_components: Mapping[str, ContextComponent] | None = None,
        moderation_rules: list[str] | None = None,
        authority_level: str = "standard",
        managed_entities: list[str] | None = None,
        language_model: Any = None,  # LanguageModel type, but avoiding circular import
    ):
        """Initializes an Archon agent.
        
        Args:
            agent_name: The name of the archon.
            act_component: The component responsible for moderation decisions.
                If None and language_model is provided, uses ArchonLLMComponent.
                If both are None, uses default ArchonActingComponent.
            context_components: Optional context components for the archon.
            moderation_rules: Optional list of rules this archon enforces.
            authority_level: The level of authority (e.g., "standard", "high", "supreme").
            managed_entities: Optional list of entity names this archon manages.
            language_model: Optional language model for LLM-powered decision making.
                If provided and act_component is None, creates an ArchonLLMComponent.
        """
        # Create appropriate acting component
        if act_component is None:
            if language_model is not None:
                act_component = ArchonLLMComponent(language_model)
            else:
                act_component = ArchonActingComponent()
            
        super().__init__(
            agent_name=agent_name,
            act_component=act_component,
            context_components=context_components,
        )
        
        # Archon-specific attributes
        self._moderation_rules = list(moderation_rules or [])
        self._authority_level = authority_level
        self._managed_entities = list(managed_entities or [])
        self._role = "moderator"  # Distinguishes from Actor's "participant" role
        self._session_state = "inactive"  # inactive, active, paused, concluded
    
    @property
    def moderation_rules(self) -> list[str]:
        """Returns the archon's moderation rules."""
        return self._moderation_rules.copy()
    
    def add_moderation_rule(self, rule: str) -> None:
        """Adds a new moderation rule.
        
        Args:
            rule: The rule to add.
        """
        if rule not in self._moderation_rules:
            self._moderation_rules.append(rule)
    
    def remove_moderation_rule(self, rule: str) -> None:
        """Removes a moderation rule.
        
        Args:
            rule: The rule to remove.
        """
        if rule in self._moderation_rules:
            self._moderation_rules.remove(rule)
    
    @property
    def authority_level(self) -> str:
        """Returns the archon's authority level."""
        return self._authority_level
    
    def set_authority_level(self, level: str) -> None:
        """Sets the archon's authority level.
        
        Args:
            level: The new authority level.
        """
        self._authority_level = level
    
    @property
    def managed_entities(self) -> list[str]:
        """Returns the entities managed by this archon."""
        return self._managed_entities.copy()
    
    def add_managed_entity(self, entity_name: str) -> None:
        """Adds an entity to be managed by this archon.
        
        Args:
            entity_name: The name of the entity to manage.
        """
        if entity_name not in self._managed_entities:
            self._managed_entities.append(entity_name)
    
    def remove_managed_entity(self, entity_name: str) -> None:
        """Removes an entity from management.
        
        Args:
            entity_name: The name of the entity to stop managing.
        """
        if entity_name in self._managed_entities:
            self._managed_entities.remove(entity_name)
    
    @property
    def role(self) -> str:
        """Returns the agent's role type (always 'moderator' for Archons)."""
        return self._role
    
    @property
    def session_state(self) -> str:
        """Returns the current session state."""
        return self._session_state
    
    def set_session_state(self, state: str) -> None:
        """Sets the session state.
        
        Args:
            state: The new session state (inactive, active, paused, concluded).
        """
        valid_states = ["inactive", "active", "paused", "concluded"]
        if state not in valid_states:
            raise ValueError(f"Invalid state {state}. Must be one of {valid_states}")
        self._session_state = state
    
    def moderate_interaction(self, participants: Sequence[str], interaction_type: str) -> str:
        """Moderates an interaction between participants.
        
        Args:
            participants: The names of the participants in the interaction.
            interaction_type: The type of interaction (e.g., "debate", "negotiation").
            
        Returns:
            The moderation action or decision.
        """
        # This is a placeholder - in practice, this would use sophisticated
        # moderation logic based on rules, context, and AI decision-making
        return f"{self.name} moderates {interaction_type} between {', '.join(participants)}"
    
    def evaluate_performance(self, entity_name: str, criteria: dict[str, Any]) -> dict[str, Any]:
        """Evaluates the performance of an entity.
        
        Args:
            entity_name: The name of the entity to evaluate.
            criteria: The evaluation criteria.
            
        Returns:
            The evaluation results.
        """
        # Placeholder implementation
        return {
            "entity": entity_name,
            "evaluator": self.name,
            "criteria": criteria,
            "score": 0.5,  # Placeholder
            "notes": "Evaluation completed by archon."
        }
    
    def get_archon_context(self) -> dict[str, Any]:
        """Returns Archon-specific context information.
        
        This method provides Archon-specific information that can be used
        by components or other systems that need to understand this agent's
        moderation role and capabilities.
        
        Returns:
            Dictionary containing archon-specific context.
        """
        return {
            "role": self._role,
            "moderation_rules": self._moderation_rules,
            "authority_level": self._authority_level,
            "managed_entities": self._managed_entities,
            "session_state": self._session_state,
            "agent_type": "Archon",
        }


class ArchonWithLogging(Archon):
    """An Archon that includes logging capabilities for debugging and monitoring."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._moderation_log: list[dict[str, Any]] = []
        self._observation_log: list[str] = []
        self._evaluation_log: list[dict[str, Any]] = []
    
    def act(self, action_spec: ActionSpec) -> str:
        """Acts and logs the moderation action for debugging."""
        action = super().act(action_spec)
        
        # Log the moderation action
        log_entry = {
            "action": action,
            "action_spec": {
                "call_to_action": action_spec.call_to_action,
                "output_type": action_spec.output_type.name,
                "options": action_spec.options,
                "tag": action_spec.tag,
            },
            "phase": self.get_phase().name,
            "session_state": self._session_state,
            "managed_entities": self._managed_entities,
        }
        self._moderation_log.append(log_entry)
        
        return action
    
    def observe(self, observation: str) -> None:
        """Observes and logs the observation for debugging."""
        super().observe(observation)
        self._observation_log.append(observation)
    
    def moderate_interaction(self, participants: Sequence[str], interaction_type: str) -> str:
        """Moderates interaction and logs the decision."""
        result = super().moderate_interaction(participants, interaction_type)
        
        # Log the moderation decision
        log_entry = {
            "participants": list(participants),
            "interaction_type": interaction_type,
            "moderation_result": result,
            "session_state": self._session_state,
        }
        self._moderation_log.append(log_entry)
        
        return result
    
    def evaluate_performance(self, entity_name: str, criteria: dict[str, Any]) -> dict[str, Any]:
        """Evaluates performance and logs the evaluation."""
        result = super().evaluate_performance(entity_name, criteria)
        self._evaluation_log.append(result)
        return result
    
    def get_last_log(self) -> dict[str, Any]:
        """Returns the most recent debugging information."""
        return {
            "agent_name": self.name,
            "agent_type": "Archon",
            "role": self._role,
            "authority_level": self._authority_level,
            "session_state": self._session_state,
            "managed_entities": self._managed_entities,
            "moderation_rules": self._moderation_rules,
            "recent_moderations": self._moderation_log[-5:],  # Last 5 moderation actions
            "recent_observations": self._observation_log[-5:],  # Last 5 observations
            "recent_evaluations": self._evaluation_log[-3:],  # Last 3 evaluations
            "current_phase": self.get_phase().name,
            "component_count": len(self.get_all_context_components()),
        }
    
    def get_full_log(self) -> dict[str, Any]:
        """Returns complete logging information."""
        return {
            "agent_name": self.name,
            "agent_type": "Archon",
            "role": self._role,
            "authority_level": self._authority_level,
            "session_state": self._session_state,
            "managed_entities": self._managed_entities,
            "moderation_rules": self._moderation_rules,
            "all_moderations": self._moderation_log,
            "all_observations": self._observation_log,
            "all_evaluations": self._evaluation_log,
            "current_phase": self.get_phase().name,
            "components": list(self.get_all_context_components().keys()),
        }
