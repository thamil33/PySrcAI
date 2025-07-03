"""Base Agent class for PySrcAI - equivalent to Concordia's Entity abstraction.

This module provides the foundational Agent class that serves as the base for
all entities in PySrcAI simulations. Unlike Concordia's Entity, this Agent class
is designed for modularity and clear separation between Actor and Archon roles.

The Agent class provides:
- Core entity interface (name, act, observe)
- Component system integration
- Memory management interface
- Language model integration
- Thread-safe state management
- Configuration-driven instantiation

Specialized classes (Actor, Archon) inherit from Agent and add role-specific
functionality while maintaining the common base interface.
"""

import abc
import functools
import threading
import types
from collections.abc import Mapping, Sequence
from typing import Any, cast, TypeVar
import dataclasses
import enum

# Type definitions
ComponentName = str
ComponentContext = str
ComponentContextMapping = Mapping[ComponentName, ComponentContext]
ComponentT = TypeVar("ComponentT", bound="BaseComponent")

_LeafT = str | int | float | None
_ValueT = _LeafT | Sequence["_ValueT"] | Mapping[str, "_ValueT"]
ComponentState = Mapping[str, _ValueT]


@enum.unique
class OutputType(enum.Enum):
    """The type of output that an agent can produce."""
    # General output types
    FREE = enum.auto()
    CHOICE = enum.auto()
    FLOAT = enum.auto()
    # Actor-specific output types
    ACTION = enum.auto()
    SPEECH = enum.auto()
    DECISION = enum.auto()
    # Archon-specific output types
    OBSERVE = enum.auto()
    MODERATE = enum.auto()
    EVALUATE = enum.auto()
    ORCHESTRATE = enum.auto()
    TERMINATE = enum.auto()


# Action type groups for validation
ACTOR_ACTION_TYPES = (
    OutputType.FREE,
    OutputType.CHOICE,
    OutputType.FLOAT,
    OutputType.ACTION,
    OutputType.SPEECH,
    OutputType.DECISION,
)

ARCHON_ACTION_TYPES = (
    OutputType.OBSERVE,
    OutputType.MODERATE,
    OutputType.EVALUATE,
    OutputType.ORCHESTRATE,
    OutputType.TERMINATE,
    OutputType.CHOICE,
)

FREE_ACTION_TYPES = (
    OutputType.FREE,
    OutputType.ACTION,
    OutputType.SPEECH,
    OutputType.OBSERVE,
    OutputType.MODERATE,
)

CHOICE_ACTION_TYPES = (
    OutputType.CHOICE,
    OutputType.DECISION,
    OutputType.TERMINATE,
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ActionSpec:
    """Specification for an action that an agent is queried for.
    
    This is the PySrcAI equivalent of Concordia's ActionSpec, adapted for
    the Actor/Archon distinction and modular architecture.
    
    Attributes:
        call_to_action: Formatted text conditioning agent response.
            {name} and {timedelta} will be inserted by the agent.
        output_type: Type of output expected from the agent.
        options: For multiple choice actions, the available options.
        tag: A tag to add to memory (e.g., action, speech, decision).
        context: Additional context for the action.
    """
    call_to_action: str
    output_type: OutputType
    options: Sequence[str] = ()
    tag: str | None = None
    context: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.output_type in CHOICE_ACTION_TYPES:
            if not self.options:
                raise ValueError('Options must be provided for CHOICE output type.')
            if len(set(self.options)) != len(self.options):
                raise ValueError('Options must not contain duplicate choices.')
        elif self.options:
            raise ValueError('Options not supported for non-CHOICE output type.')
        object.__setattr__(self, 'options', tuple(self.options))

    def validate(self, action: str) -> None:
        """Validates the specified action against the action spec.
        
        Args:
            action: The action to validate.
            
        Raises:
            ValueError: If the action is invalid.
        """
        if self.output_type == OutputType.FREE:
            return
        elif self.output_type in CHOICE_ACTION_TYPES:
            if action not in self.options:
                raise ValueError(f'Action {action!r} is not one of {self.options!r}.')
        elif self.output_type == OutputType.FLOAT:
            try:
                float(action)
            except ValueError:
                raise ValueError(f'Action {action!r} is not a valid float.') from None
        # Add other validations as needed


class Phase(enum.Enum):
    """Phases of an agent's lifecycle during action/observation processing.
    
    This maintains compatibility with Concordia's phase system while being
    adapted for PySrcAI's architecture.
    """
    READY = enum.auto()
    PRE_ACT = enum.auto()
    POST_ACT = enum.auto()
    PRE_OBSERVE = enum.auto()
    POST_OBSERVE = enum.auto()
    UPDATE = enum.auto()

    @functools.cached_property
    def successors(self) -> tuple["Phase", ...]:
        """Returns the phases which may follow the current phase."""
        match self:
            case Phase.READY:
                return (Phase.PRE_ACT, Phase.PRE_OBSERVE)
            case Phase.PRE_ACT:
                return (Phase.POST_ACT,)
            case Phase.POST_ACT:
                return (Phase.UPDATE,)
            case Phase.PRE_OBSERVE:
                return (Phase.POST_OBSERVE,)
            case Phase.POST_OBSERVE:
                return (Phase.UPDATE,)
            case Phase.UPDATE:
                return (Phase.READY,)

    def check_successor(self, successor: "Phase") -> None:
        """Raises ValueError if successor is not a valid next phase."""
        if successor not in self.successors:
            raise ValueError(f"The transition from {self} to {successor} is invalid.")


class BaseComponent:
    """Base class for all agent components in PySrcAI.
    
    Components are modular pieces of functionality that can be attached to
    agents to provide specific capabilities. This follows Concordia's component
    pattern but is adapted for PySrcAI's modular architecture.
    """
    
    _agent: "Agent | None" = None

    def set_agent(self, agent: "Agent") -> None:
        """Sets the agent that this component belongs to.
        
        Args:
            agent: The agent that this component belongs to.
            
        Raises:
            RuntimeError: If the agent is already set to a different agent.
        """
        if self._agent is not None and self._agent != agent:
            raise RuntimeError("Agent is already set to a different agent.")
        self._agent = agent

    def get_agent(self) -> "Agent":
        """Returns the agent that this component belongs to.
        
        Raises:
            RuntimeError: If the agent is not set.
        """
        if self._agent is None:
            raise RuntimeError("Agent is not set.")
        return self._agent

    def get_state(self) -> ComponentState:
        """Returns the state of the component for serialization/persistence."""
        return {}

    def set_state(self, state: ComponentState) -> None:
        """Sets the state of the component from serialized data."""
        pass


class ContextComponent(BaseComponent):
    """A component that provides context during agent processing phases.
    
    Context components are called during various phases of agent execution
    to provide relevant information for decision-making.
    """

    def pre_act(self, action_spec: ActionSpec) -> str:
        """Returns relevant information for the agent to act.
        
        Args:
            action_spec: The action specification for the action attempt.
            
        Returns:
            Relevant context information for acting.
        """
        return ""

    def post_act(self, action_attempt: str) -> str:
        """Processes the action attempted by the agent.
        
        Args:
            action_attempt: The action that the agent attempted.
            
        Returns:
            Any information that needs to bubble up to the agent.
        """
        return ""

    def pre_observe(self, observation: str) -> str:
        """Processes an observation before the agent's main observation handling.
        
        Args:
            observation: The observation that the agent received.
            
        Returns:
            Relevant context for processing the observation.
        """
        return ""

    def post_observe(self) -> str:
        """Provides context after observation processing is complete.
        
        Returns:
            Any information that needs to bubble up to the agent.
        """
        return ""

    def update(self) -> None:
        """Updates the component's internal state.
        
        Called after all components have processed an action or observation.
        """
        pass


class ActingComponent(BaseComponent, metaclass=abc.ABCMeta):
    """A privileged component that makes action decisions for the agent.
    
    Each agent must have exactly one ActingComponent that is responsible
    for making the final decision on what action to take.
    """

    @abc.abstractmethod
    def get_action_attempt(
        self,
        context: ComponentContextMapping,
        action_spec: ActionSpec,
    ) -> str:
        """Decides the action for the agent.
        
        Args:
            context: Context from all the agent's components.
            action_spec: The specification for the action.
            
        Returns:
            The action that the agent should attempt.
        """
        raise NotImplementedError()


class Agent(metaclass=abc.ABCMeta):
    """Base class for all agents in PySrcAI.
    
    This is the foundational class for all autonomous entities in PySrcAI
    simulations. It provides the core interface and functionality that both
    Actors (simulation participants) and Archons (simulation moderators)
    inherit from.
    
    The Agent class is designed to be:
    - Modular: Uses a component system for extensibility
    - Thread-safe: Handles concurrent access safely
    - Configurable: Can be instantiated from configuration files
    - Memory-aware: Integrates with centralized memory management
    - LLM-ready: Designed for AI-powered decision making
    
    Key differences from Concordia's Entity:
    - Clearer separation of concerns through Actor/Archon hierarchy
    - Centralized memory management
    - Enhanced component system
    - Better configuration support
    """

    def __init__(
        self,
        agent_name: str,
        act_component: ActingComponent,
        context_components: Mapping[str, ContextComponent] | None = None,
    ):
        """Initializes the agent.
        
        Args:
            agent_name: The name of the agent.
            act_component: The component responsible for making action decisions.
            context_components: Optional context components for the agent.
        """
        self._agent_name = agent_name
        self._control_lock = threading.Lock()
        self._phase_lock = threading.Lock()
        self._phase = Phase.READY
        
        # Set up the acting component
        self._act_component = act_component
        self._act_component.set_agent(self)
        
        # Set up context components
        self._context_components = dict(context_components or {})
        for component in self._context_components.values():
            component.set_agent(self)

    @functools.cached_property
    def name(self) -> str:
        """The name of the agent."""
        return self._agent_name

    def get_phase(self) -> Phase:
        """Returns the current phase of the agent."""
        with self._phase_lock:
            return self._phase

    def _set_phase(self, phase: Phase) -> None:
        """Sets the current phase of the agent."""
        with self._phase_lock:
            self._phase.check_successor(phase)
            self._phase = phase

    def get_component(
        self,
        name: str,
        *,
        type_: type[ComponentT] = BaseComponent,
    ) -> ComponentT:
        """Returns the component with the given name.
        
        Args:
            name: The name of the component to fetch.
            type_: If passed, the returned component will be cast to this type.
            
        Returns:
            The requested component.
            
        Raises:
            KeyError: If the component is not found.
        """
        component = self._context_components[name]
        return cast(ComponentT, component)

    def get_act_component(self) -> ActingComponent:
        """Returns the acting component."""
        return self._act_component

    def get_all_context_components(self) -> Mapping[str, ContextComponent]:
        """Returns all context components."""
        return types.MappingProxyType(self._context_components)

    def _parallel_call(
        self,
        method_name: str,
        *args,
    ) -> ComponentContextMapping:
        """Calls the named method in parallel on all context components.
        
        Args:
            method_name: The name of the method to call.
            *args: The arguments to pass to the method.
            
        Returns:
            A mapping of component name to the result of the method call.
        """
        # For now, implement sequentially. Can optimize with actual parallel execution later.
        results: dict[str, str] = {}
        
        for name, component in self._context_components.items():
            method = getattr(component, method_name)
            results[name] = method(*args)
            
        return types.MappingProxyType(results)

    def act(self, action_spec: ActionSpec) -> str:
        """Returns the agent's intended action given the action spec.
        
        This method orchestrates the entire action process:
        1. Pre-act phase: Gather context from all components
        2. Action decision: Use the acting component to decide action
        3. Post-act phase: Inform components of the action
        4. Update phase: Allow components to update their state
        
        Args:
            action_spec: The specification of the action requested.
            
        Returns:
            The agent's intended action.
        """
        with self._control_lock:
            # Pre-act phase
            self._set_phase(Phase.PRE_ACT)
            contexts = self._parallel_call('pre_act', action_spec)
            
            # Get action from acting component
            action_attempt = self._act_component.get_action_attempt(contexts, action_spec)
            
            # Validate the action
            action_spec.validate(action_attempt)
            
            # Post-act phase
            self._set_phase(Phase.POST_ACT)
            self._parallel_call('post_act', action_attempt)
            
            # Update phase
            self._set_phase(Phase.UPDATE)
            self._parallel_call('update')
            
            # Return to ready state
            self._set_phase(Phase.READY)
            
            return action_attempt

    def observe(self, observation: str) -> None:
        """Informs the agent of an observation.
        
        This method orchestrates the observation process:
        1. Pre-observe phase: Prepare components for observation
        2. Observation processing: Components process the observation
        3. Post-observe phase: Components provide final context
        4. Update phase: Components update their state
        
        Args:
            observation: The observation for the agent to process.
        """
        with self._control_lock:
            # Pre-observe phase
            self._set_phase(Phase.PRE_OBSERVE)
            self._parallel_call('pre_observe', observation)
            
            # Post-observe phase
            self._set_phase(Phase.POST_OBSERVE)
            self._parallel_call('post_observe')
            
            # Update phase
            self._set_phase(Phase.UPDATE)
            self._parallel_call('update')
            
            # Return to ready state
            self._set_phase(Phase.READY)


class AgentWithLogging(Agent):
    """An agent interface that includes logging capabilities.
    
    This extends the base Agent class with debugging and logging functionality,
    useful for development, testing, and monitoring agent behavior.
    """

    @abc.abstractmethod
    def get_last_log(self) -> dict[str, Any]:
        """Returns debugging information in the form of a dictionary."""
        raise NotImplementedError()


# Default action specifications
DEFAULT_CALL_TO_ACTION = (
    'What would {name} do next? '
    'Give a specific activity. '
    'If the selected action has a direct or indirect object then it '
    'must be specified explicitly.'
)

DEFAULT_ACTION_SPEC = ActionSpec(
    call_to_action=DEFAULT_CALL_TO_ACTION,
    output_type=OutputType.FREE,
    tag='action',
)

DEFAULT_CALL_TO_SPEECH = (
    'Given the above, what is {name} likely to say next? Respond in '
    'the format `{name} -- "..."` For example, '
    'Alice -- "Hello! How are you today?", '
    'Bob -- "I think we should consider this carefully", or '
    'Moderator -- "Let\'s move to the next topic".\n'
)

DEFAULT_SPEECH_ACTION_SPEC = ActionSpec(
    call_to_action=DEFAULT_CALL_TO_SPEECH,
    output_type=OutputType.SPEECH,
    tag='speech',
)


# Convenience functions for creating action specs
def free_action_spec(**kwargs) -> ActionSpec:
    """Returns an action spec with output type FREE."""
    return ActionSpec(output_type=OutputType.FREE, **kwargs)

def choice_action_spec(**kwargs) -> ActionSpec:
    """Returns an action spec with output type CHOICE."""
    return ActionSpec(output_type=OutputType.CHOICE, **kwargs)

def float_action_spec(**kwargs) -> ActionSpec:
    """Returns an action spec with output type FLOAT."""
    return ActionSpec(output_type=OutputType.FLOAT, **kwargs)

def speech_action_spec(**kwargs) -> ActionSpec:
    """Returns an action spec with output type SPEECH."""
    return ActionSpec(output_type=OutputType.SPEECH, **kwargs)
