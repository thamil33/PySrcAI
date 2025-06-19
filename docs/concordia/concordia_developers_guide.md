# Concordia Developer's Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Core Architecture](#core-architecture)
3. [Component System](#component-system)
4. [Entity Framework](#entity-framework)
5. [Phase-Based Execution](#phase-based-execution)
6. [Memory Systems](#memory-systems)
7. [Language Model Integration](#language-model-integration)
8. [Environment and Simulation](#environment-and-simulation)
9. [Prefabs and Configuration](#prefabs-and-configuration)
10. [Development Patterns](#development-patterns)
11. [Testing and Debugging](#testing-and-debugging)
12. [Extension Points](#extension-points)

## Introduction

Concordia is a sophisticated framework for building generative agent-based simulations, designed with modularity and extensibility at its core. This developer's guide provides comprehensive documentation for understanding, extending, and contributing to the Concordia framework.

### Design Philosophy

Concordia's architecture is built on several key principles:
- **Component-Based Design**: Complex behaviors emerge from simple, reusable components
- **Composition over Inheritance**: Functionality is assembled rather than inherited
- **Phase-Based Processing**: Ordered execution ensures predictable state management
- **Language Model Integration**: LLMs serve as the cognitive core of agent behavior
- **Extensibility**: New functionality can be added through well-defined interfaces

## Core Architecture

### Entity System

The foundation of Concordia is the entity system, where each participant in a simulation is represented as an `Entity`. The system defines two fundamental capabilities:

#### Acting
Entities determine and execute actions based on their current state and environment through a structured process:
```python
def act(self, action_spec: entity.ActionSpec) -> str:
    # Gather context from all components
    # Process through acting component
    # Return action attempt
```

#### Observing
Entities process information from the environment to update their internal state:
```python
def observe(self, observation: str) -> None:
    # Process observation through components
    # Update internal state
    # Commit changes
```

### Component-Based Architecture

Components are self-contained, reusable Python classes that encapsulate specific pieces of an entity's state or behavior. They provide several benefits:

- **Modularity**: Independent development and testing
- **Reusability**: Common behaviors shared across agent types
- **Flexibility**: Runtime composition and reconfiguration
- **Maintainability**: Clear separation of concerns

## Component System

### Base Component Hierarchy

```python
class BaseComponent:
    """Foundation for all components"""
    def set_entity(self, entity: EntityWithComponents) -> None
    def get_entity(self) -> EntityWithComponents
    def get_state(self) -> ComponentState
    def set_state(self, state: ComponentState) -> None

class ContextComponent(BaseComponent):
    """Provides context during lifecycle phases"""
    def pre_act(self, action_spec: ActionSpec) -> str
    def post_act(self, action_attempt: str) -> None
    def pre_observe(self, observation: str) -> str
    def post_observe(self) -> str
    def update(self) -> None

class ActingComponent(BaseComponent):
    """Determines agent actions"""
    def get_action_attempt(
        self, 
        contexts: ComponentContextMapping, 
        action_spec: ActionSpec
    ) -> str
```

### Standard Component Library

#### Memory Components
- **AssociativeMemory**: Uses memory banks for storage and retrieval
- **ListMemory**: Simple list-based memory storage
- **ObservationToMemory**: Automatically stores observations
- **LastNObservations**: Provides recent observation history

#### Cognitive Components
- **Plan**: Manages goal-oriented planning
- **QuestionOfRecentMemories**: Reflects on recent experiences
- **SituationPerception**: Analyzes current context
- **SelfPerception**: Maintains identity model
- **PersonBySituation**: Determines contextually appropriate actions

#### Utility Components
- **Constant**: Provides static string content
- **Instructions**: Specialized instructions for agents
- **ReportFunction**: Dynamic content from functions
- **ActionSpecIgnored**: Cacheable component outputs

### Creating Custom Components

```python
class CustomComponent(entity_component.ContextComponent):
    def __init__(self, custom_param: str):
        super().__init__()
        self._custom_param = custom_param
    
    def pre_act(self, action_spec: entity.ActionSpec) -> str:
        # Provide context for action decisions
        return f"Custom context: {self._custom_param}"
    
    def update(self) -> None:
        # Update component state
        pass
```

## Entity Framework

### EntityAgent Implementation

The `EntityAgent` class serves as the container for components:

```python
class EntityAgent(entity_component.EntityWithComponents):
    def __init__(
        self,
        agent_name: str,
        act_component: entity_component.ActingComponent,
        context_processor: entity_component.ContextProcessorComponent = None,
        context_components: Mapping[str, entity_component.ContextComponent] = None,
    ):
        # Initialize components and set entity references
        # Set up phase management and thread safety
```

### Key Features

- **Named Identity**: Each agent has a unique identifier
- **Component Management**: Maintains collections of behavioral components
- **Thread Safety**: Uses locks for concurrent access protection
- **Phase Tracking**: Manages execution state through phases

### Entity Types

#### Agents
Individual autonomous actors whose behaviors are the subject of study:
- Perceive through components
- Reason using language models
- Execute actions in environment
- Exhibit emergent behavior through component interaction

#### Game Master
Special orchestrating entity that manages the simulation:
- Controls environment state
- Enforces simulation rules
- Delivers observations to agents
- Built from components like regular agents

## Phase-Based Execution

### Phase System

Concordia uses a phase system to manage information flow and ensure predictable state transitions:

```python
class Phase(enum.Enum):
    READY = enum.auto()         # Ready for observe or act
    PRE_ACT = enum.auto()       # Gathering action context
    POST_ACT = enum.auto()      # Processing action aftermath  
    PRE_OBSERVE = enum.auto()   # Preparing for observation
    POST_OBSERVE = enum.auto()  # Processing observation
    UPDATE = enum.auto()        # Committing state changes
```

### Phase Transitions

```
READY → PRE_ACT → POST_ACT → UPDATE → READY
READY → PRE_OBSERVE → POST_OBSERVE → UPDATE → READY
```

### Implementation Details

```python
def act(self, action_spec: entity.ActionSpec) -> str:
    with self._control_lock:
        self._set_phase(Phase.PRE_ACT)
        contexts = self._parallel_call_('pre_act', action_spec)
        
        self._context_processor.pre_act(contexts)
        action_attempt = self._act_component.get_action_attempt(contexts, action_spec)
        
        self._set_phase(Phase.POST_ACT)
        self._parallel_call_('post_act', action_attempt)
        
        self._set_phase(Phase.UPDATE)
        self._parallel_call_('update')
        
        self._set_phase(Phase.READY)
        return action_attempt
```

## Memory Systems

### Associative Memory

The associative memory system enables agents to store, retrieve, and reflect on experiences:

#### Memory Types

**Formative Memory**
- Core identity and background information
- Set during initialization
- Rarely changes during simulation
- Defines personality and foundational beliefs

**Episodic Memory**
- Records of events and observations
- Tagged with metadata (time, importance)
- Retrieved based on relevance
- Enables learning and adaptation

#### Memory Operations

```python
# Store memories
memory.add("I had a pleasant conversation with Alice.")

# Retrieve similar memories
relevant_memories = memory.retrieve_similar("talking with friends", limit=5)

# Scan with custom criteria
memories = memory.scan(lambda x: "Alice" in x)
```

### Memory Bank Integration

```python
class AssociativeMemory(entity_component.ContextComponent):
    def __init__(self, memory_bank: basic_associative_memory.AssociativeMemoryBank):
        self._memory_bank = memory_bank
        self._add_buffer = []
    
    def pre_act(self, action_spec: entity.ActionSpec) -> str:
        # Retrieve relevant memories for context
        return self._memory_bank.retrieve_recent(limit=10)
    
    def update(self) -> None:
        # Commit buffered memories
        for memory in self._add_buffer:
            self._memory_bank.add(memory)
        self._add_buffer.clear()
```

## Language Model Integration

### Language Model Interface

Concordia provides a standardized interface for various language models:

```python
class LanguageModel(abc.ABC):
    @abc.abstractmethod
    def sample_text(
        self,
        prompt: str,
        max_tokens: int = 100,
        terminators: Collection[str] = (),
    ) -> str:
        """Generate text from prompt"""
```

### Supported Backends

#### Cloud APIs
- **OpenRouter**: Via `openrouter_model.py`
- **Vertex AI**: For Google Cloud integration

#### Local Models
- **Ollama**: Local LLM serving
- **LMStudio**: Planned local API support
- **PyTorch**: Direct model integration

#### Testing
- **MockModel**: Deterministic responses for testing

### Usage in Components

```python
class ConcatActComponent(entity_component.ActingComponent):
    def __init__(self, model: language_model.LanguageModel):
        self._model = model
    
    def get_action_attempt(
        self, 
        contexts: ComponentContextMapping, 
        action_spec: ActionSpec
    ) -> str:
        prompt = self._build_prompt(contexts, action_spec)
        return self._model.sample_text(prompt)
```

## Environment and Simulation

### Game Clock

The `GameClock` orchestrates simulation execution:

```python
class GameClock:
    def __init__(self, entities: Sequence[entity.Entity]):
        self._entities = entities
    
    def advance(self) -> None:
        # Update all entities
        # Manage time progression
        # Handle event scheduling
```

### Engine System

The `Engine` class provides the simulation framework:

```python
class Engine(abc.ABC):
    @abc.abstractmethod
    def make_observation(
        self,
        game_master: entity.Entity,
        entity: entity.Entity,
    ) -> str:
        """Generate observation for entity"""
    
    @abc.abstractmethod
    def next_acting(
        self,
        game_master: entity.Entity,
        entities: Sequence[entity.Entity],
    ) -> entity.Entity:
        """Determine next acting entity"""
```

### Scene Management

Scenes define simulation structure and rules:
- Environment description and rules
- Initial conditions and setup
- Victory/termination conditions
- Interaction constraints

## Prefabs and Configuration

### Prefab System

Prefabs provide pre-configured entity setups:

```python
@dataclasses.dataclass
class Entity(prefab_lib.Prefab):
    description: str
    params: Mapping[str, str]
    
    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        # Configure and return entity
```

### Standard Prefabs

#### Basic Entity
Uses the "three key questions" approach:
1. "What situation am I in?"
2. "What kind of person am I?"
3. "What would a person like me do in this situation?"

Components:
- Instructions
- ObservationToMemory
- LastNObservations
- SituationPerception
- SelfPerception
- PersonBySituation
- AllSimilarMemories

#### Basic With Plan
Extends Basic Entity with planning capabilities:
- Adds Plan component for goal-oriented behavior

#### Minimal Entity
Lightweight configuration:
- Essential components only
- Configurable goals
- Extensible component list

### Creating Custom Prefabs

```python
@dataclasses.dataclass
class CustomEntity(prefab_lib.Prefab):
    description: str = "Custom agent configuration"
    params: Mapping[str, str] = dataclasses.field(default_factory=dict)
    
    def build(self, model, memory_bank):
        # Configure custom component combination
        components = {
            'memory': CustomMemoryComponent(memory_bank),
            'perception': CustomPerceptionComponent(),
            'action': CustomActionComponent(model),
        }
        
        return entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=self.params['name'],
            act_component=components['action'],
            context_components=components,
        )
```

## Development Patterns

### Component Design Principles

1. **Single Responsibility**: Each component has one clear purpose
2. **Stateless When Possible**: Minimize internal state complexity
3. **Phase Awareness**: Respect the phase system for predictable behavior
4. **Error Handling**: Graceful degradation on failures
5. **Documentation**: Clear interfaces and behavior documentation

### Best Practices

#### Component Development
```python
class WellDesignedComponent(entity_component.ContextComponent):
    """Clear documentation of component purpose and behavior."""
    
    def __init__(self, required_param: str, optional_param: str = "default"):
        """Document parameters and their purposes."""
        super().__init__()
        self._required_param = required_param
        self._optional_param = optional_param
        self._internal_state = {}
    
    def pre_act(self, action_spec: entity.ActionSpec) -> str:
        """Provide context for action decisions."""
        # Check entity phase if needed
        if self.get_entity().get_phase() != entity_component.Phase.PRE_ACT:
            raise ValueError("Invalid phase for pre_act")
        
        # Generate context
        return f"Context based on {self._required_param}"
    
    def get_state(self) -> entity_component.ComponentState:
        """Return serializable state for persistence."""
        return {
            'internal_state': self._internal_state,
            'required_param': self._required_param,
        }
    
    def set_state(self, state: entity_component.ComponentState) -> None:
        """Restore from serialized state."""
        self._internal_state = state.get('internal_state', {})
        # Note: Don't restore constructor parameters
```

#### Entity Configuration
```python
def create_sophisticated_agent(
    name: str,
    personality: str,
    goals: list[str],
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Create a complex agent with multiple capabilities."""
    
    # Core components
    memory = agent_components.memory.AssociativeMemory(memory_bank)
    instructions = agent_components.instructions.Instructions(
        agent_name=name,
        instructions=f"You are {personality}"
    )
    
    # Cognitive components
    plan = agent_components.plan.Plan(
        model=model,
        goals=goals,
    )
    
    # Acting component
    acting = agent_components.concat_act_component.ConcatActComponent(
        model=model,
        logging_channel=logging_lib.LoggingChannel.ACTING,
    )
    
    # Assemble entity
    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=acting,
        context_components={
            'memory': memory,
            'instructions': instructions,
            'plan': plan,
        }
    )
```

## Testing and Debugging

### Mock Components

Use mock components for predictable testing:

```python
class MockLanguageModel(language_model.LanguageModel):
    def __init__(self, responses: list[str]):
        self._responses = responses
        self._call_count = 0
    
    def sample_text(self, prompt: str, **kwargs) -> str:
        response = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return response
```

### Component Testing

```python
def test_custom_component():
    # Create mock entity
    entity = create_test_entity()
    
    # Create component
    component = CustomComponent("test_param")
    component.set_entity(entity)
    
    # Test phase behavior
    entity._set_phase(entity_component.Phase.PRE_ACT)
    context = component.pre_act(entity.DEFAULT_ACTION_SPEC)
    
    assert "test_param" in context
```

### Logging and Debugging

Entities with logging provide debugging capabilities:

```python
# Access last log entries
logs = agent.get_last_log()
for entry in logs:
    print(f"{entry.timestamp}: {entry.message}")

# Component-specific logging
class DebuggingComponent(entity_component.ComponentWithLogging):
    def pre_act(self, action_spec):
        self.log("Processing action spec", action_spec)
        return super().pre_act(action_spec)
```

## Extension Points

### Custom Language Models

Implement the `LanguageModel` interface:

```python
class CustomLanguageModel(language_model.LanguageModel):
    def __init__(self, api_endpoint: str, api_key: str):
        self._endpoint = api_endpoint
        self._key = api_key
    
    def sample_text(self, prompt: str, **kwargs) -> str:
        # Implement API call
        response = self._call_api(prompt, **kwargs)
        return response['text']
```

### Custom Engines

Extend the `Engine` class for specialized simulations:

```python
class TurnBasedEngine(Engine):
    def __init__(self, turn_order: list[str]):
        self._turn_order = turn_order
        self._current_turn = 0
    
    def next_acting(self, game_master, entities):
        current_name = self._turn_order[self._current_turn]
        self._current_turn = (self._current_turn + 1) % len(self._turn_order)
        
        for entity in entities:
            if entity.name == current_name:
                return entity
        raise ValueError(f"Entity {current_name} not found")
```

### Memory Bank Extensions

Customize memory storage and retrieval:

```python
class VectorMemoryBank(basic_associative_memory.AssociativeMemoryBank):
    def __init__(self, embedding_model):
        self._embedding_model = embedding_model
        self._memories = []
        self._embeddings = []
    
    def add(self, memory: str, metadata: dict = None):
        embedding = self._embedding_model.encode(memory)
        self._memories.append((memory, metadata))
        self._embeddings.append(embedding)
    
    def retrieve_similar(self, query: str, limit: int = 5):
        query_embedding = self._embedding_model.encode(query)
        similarities = cosine_similarity([query_embedding], self._embeddings)[0]
        top_indices = similarities.argsort()[-limit:][::-1]
        return [self._memories[i][0] for i in top_indices]
```

## Conclusion

This developer's guide provides a comprehensive foundation for working with the Concordia framework. The component-based architecture, phase system, and language model integration create a powerful foundation for building sophisticated agent-based simulations.

Key takeaways:
- Components are the building blocks of agent behavior
- Phases ensure predictable execution and state management
- Language models provide the cognitive core of agent reasoning
- Prefabs accelerate development with proven configurations
- Extension points enable customization for specific use cases

## Additional Resources

For more detailed information, refer to:
- **API Documentation**: Complete auto-generated API reference in `concordia/.api_docs/` (JSON format for LLM consumption)
- **Inline documentation**: Docstrings within the source code files in `concordia/`
- **Example implementations**: Prefabs and components in `concordia/prefabs/` and `concordia/components/`
- **Type definitions**: Interface specifications in `concordia/typing/`
- **Test files**: Usage examples in files ending with `_test.py`
- **Integration tests**: `concordia/concordia_integration_test.py`
