# Concordia Framework Developer's Guide

## Introduction

Concordia is a powerful framework for creating multi-agent simulations and environments. It provides a flexible architecture for building complex systems where multiple AI agents can interact with each other and their environment according to defined rules and scenarios.

This guide provides developers with the knowledge and tools needed to effectively build simulations using the Concordia framework. It explores the core architectural components, design patterns, and best practices for creating robust multi-agent systems.

## Core Architecture

Concordia's architecture is built around these key components:

### 1. Agents and Entities

Agents (entities) are the core actors within a Concordia simulation. They observe their environment, make decisions, and take actions. The main entity classes are:

- `EntityAgent`: The base agent class that uses components to define its functionality
- `EntityAgentWithLogging`: An extension that adds comprehensive logging

Entities are composed of multiple components that determine their capabilities:

```
EntityAgent
├── ActComponent (required)
├── ContextProcessor (optional)
└── ContextComponents (optional)
    ├── Memory Components
    ├── Observation Components
    └── Decision Components
```

### 2. Component System

The component system allows for modular and reusable behavior definitions:

- **ActingComponent**: Defines how an entity takes actions
- **ContextProcessorComponent**: Processes contexts for agent behavior
- **ContextComponent**: Provides specific capabilities to agents
  - **Memory Components**: Store and retrieve information
  - **Observation Components**: Process environmental inputs
  - **Decision Components**: Determine agent behavior

### 3. Environment and Engine

The simulation environment orchestrates entity interactions:

- **Engine**: Base class that defines the simulation loop
- **Basic Engine**: A straightforward implementation for most simulations
- **Sequential Engine**: Processes entity actions in sequence

The engine is responsible for:

- Determining which entity acts next
- Creating observations for entities
- Resolving the effects of entity actions
- Checking termination conditions

### 4. Simulation

The Simulation class ties everything together:

- Manages game masters and entities
- Configures the environment
- Orchestrates the simulation flow
- Generates output logs

### 5. Language Models

Language models are the intelligence behind agent behavior:

- **LanguageModel**: Abstract interface for LLM integration
- **OpenRouter/LMStudio/etc**: Concrete implementations for specific services

### 6. Memory Systems

Memory allows agents to have persistent knowledge:

- **AssociativeMemoryBank**: Stores and retrieves semantically similar information
- **FormativeMemories**: Defines core identity and knowledge of agents

## Getting Started

### Creating a Basic Simulation

Here's a minimal example of setting up a Concordia simulation:

```python
from concordia import language_model
from concordia.prefabs.simulation import generic as simulation_prefab
from concordia.typing import prefab as prefab_lib

# 1. Configure language model
model = language_model.OpenRouterModel(api_key="your_api_key")

# 2. Setup embedding function
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2").encode

# 3. Define simulation configuration
config = simulation_prefab.Config(
    name="Simple Simulation",
    premise="A simple conversation between two agents",
    instances=[
        prefab_lib.InstanceConfig(
            prefab='agent_entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': "Agent1",
                'goal': "To have a productive conversation",
                'context': "You are helpful and informative"
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='agent_entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': "Agent2",
                'goal': "To learn new information",
                'context': "You are curious and ask thoughtful questions"
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='generic_game_master',
            role=prefab_lib.Role.GAME_MASTER,
            params={
                'name': "GameMaster",
                'context': "You facilitate a productive conversation"
            },
        ),
    ],
)

# 4. Create and run simulation
simulation = simulation_prefab.Simulation(
    config=config,
    model=model,
    embedder=embedder,
)

# 5. Run the simulation
html_output = simulation.play(max_steps=10)
```

## Working with Entities

### Entity Components

Entities in Concordia are built using a component architecture. Here's how to create a custom entity with specific components:

```python
from concordia.agents import entity_agent
from concordia.components.agent import act, context_processors

# Create a basic acting component
act_component = act.BasicActComponent(model)

# Create context components
memory_component = memory.MemoryComponent(memory_bank)
observation_component = observation.ObservationComponent()

# Build the entity
agent = entity_agent.EntityAgent(
    agent_name="CustomAgent",
    act_component=act_component,
    context_processor=context_processors.BasicContextProcessor(),
    context_components={
        "memory": memory_component,
        "observation": observation_component,
    }
)
```

### Entity Lifecycle

Entities follow a defined phase lifecycle during simulation:

1. **READY**: Initial state, ready to act or observe
2. **PRE_ACT/PRE_OBSERVE**: Preparing to act/observe
3. **POST_ACT/POST_OBSERVE**: Processing after act/observe
4. **UPDATE**: Updating internal state
5. Return to **READY**

This cycle ensures components are invoked in the correct order.

## Scene-Based Simulation Design

Concordia supports scene-based designs where the simulation progresses through distinct scenes:

### Scene Definition

```python
from concordia.typing import scene as scene_lib

scenes = [
    scene_lib.SceneSpec(
        scene_type=scene_lib.SceneType.CHOICE,
        name="Opening Statements",
        description="Each entity presents their opening argument"
    ),
    scene_lib.SceneSpec(
        scene_type=scene_lib.SceneType.FREE,
        name="Discussion",
        description="Entities debate the topic freely"
    ),
]
```

### Scene Tracking

The SceneTracker component automatically manages scene transitions:

```python
from concordia.components.game_master import scene_tracker

scene_tracker_component = scene_tracker.SceneTracker(
    scenes=scenes,
    transition_condition=scene_tracker.CompletedTurnsCondition(1)
)
```

## Memory Systems

### Associative Memory

The associative memory system uses semantic similarity to store and retrieve information:

```python
from concordia.associative_memory import basic_associative_memory

memory_bank = basic_associative_memory.AssociativeMemoryBank(
    sentence_embedder=embedder
)

# Adding memories
memory_bank.add("The sky is blue")
memory_bank.add("Water is wet")

# Retrieving related memories
memories = memory_bank.retrieve_associative("clouds in the atmosphere", k=2)
```

### Formative Memories

Formative memories define the core identity and knowledge of agents:

```python
from concordia.associative_memory import formative_memories

formative = formative_memories.FormativeMemoryFactory(
    identity="You are a helpful assistant",
    facts=["The Earth is round", "Water freezes at 0°C"],
    relationships=["You have a cordial relationship with User"],
).create_memories()

memory_bank.extend(formative)
```

## Game Masters

The Game Master orchestrates the simulation:

```python
from concordia.prefabs.game_master import dialogic_and_dramaturgic

game_master = dialogic_and_dramaturgic.DialogicAndDramaturgicGameMaster(
    model=model,
    memory_bank=game_master_memory_bank,
    components=[
        scene_tracker.SceneTracker(scenes=scenes),
        turn_taking.TurnTaking(),
    ]
)
```

Game Masters are responsible for:

1. Determining which entity acts next
2. Creating observations for entities
3. Resolving the effects of entity actions
4. Managing scene transitions

## Advanced Features

### Clocks

Clocks provide time management within simulations:

```python
from concordia.clocks import game_clock

clock = game_clock.GameClock()
clock.advance(1)  # Advance time by 1 unit
```

### Thought Chains

Thought chains enable complex reasoning:

```python
from concordia.thought_chains import thought_chains

reasoning = thought_chains.chain_of_thought(
    model,
    "What is the capital of France?",
    "Let me think step by step"
)
```

## Best Practices from Geo_Mod Example

Based on the `geo_mod` implementation, here are some best practices for Concordia development:

### 1. Centralized Configuration

Keep configurations centralized and import them where needed:

```python
# Define in one place
WORD_LIMITS = {
    'opening_statements': {'min': 50, 'max': 100},
    'rebuttals': {'min': 50, 'max': 100},
}

# Reference in multiple places
config = {..., 'word_limits': WORD_LIMITS}
```

### 2. Scene-Based Design

Structure complex simulations as a series of well-defined scenes:

```python
SCENARIO_SCENES = [
    scene_lib.SceneSpec(
        scene_type=scene_lib.SceneType.CHOICE,
        name="Opening Statements",
        description="Entities present their opening arguments"
    ),
    # Additional scenes...
]
```

### 3. Entity Specialization

Create specialized entities for different roles:

- **Nation Entity**: Represents countries with specific goals
- **Moderator Entity**: Facilitates discussions and maintains order

### 4. Memory Integration

Integrate memory systems effectively:

```python
# Create memory factory
memory_factory = FormativeMemoryFactory(
    identity=identity_text,
    facts=context_facts,
    relationships=relationship_texts
)

# Build memories
memories = memory_factory.create_memories()

# Add to memory bank
memory_bank.extend(memories)
```

### 5. Comprehensive Logging

Implement thorough logging for debugging:

```python
agent = EntityAgentWithLogging(
    agent_name=name,
    act_component=act_component,
    context_processor=processor,
    context_components=components
)
```

### 6. HTML Results Generation

Generate rich HTML reports for simulation results:

```python
html_results = simulation.play()
with open("simulation_results.html", "w") as f:
    f.write(html_results)
```

## Developing a Modular Debate Engine

For creating a modular debate engine (as in the current development focus), consider:

### 1. Component Abstraction

Create abstract components that can be reused across different debate formats:

```python
class DebateActingComponent(ActingComponent):
    """Base component for debate actions"""
    
class ArgumentComponent(ContextComponent):
    """Component for formulating arguments"""
```

### 2. Flexible Scene Definitions

Design scenes that can adapt to different debate formats:

```python
def create_debate_scenes(format_type, time_limits):
    """Factory function for creating debate scenes"""
    if format_type == "parliamentary":
        return [
            # Parliamentary debate scenes
        ]
    elif format_type == "oxford":
        return [
            # Oxford style debate scenes
        ]
```

### 3. Pluggable Evaluation Metrics

Create modular evaluation components:

```python
class ArgumentEvaluator(ContextComponent):
    """Base evaluator for arguments"""
    
class FactualAccuracyEvaluator(ArgumentEvaluator):
    """Evaluates factual accuracy of arguments"""
    
class PersuasivenessEvaluator(ArgumentEvaluator):
    """Evaluates persuasiveness of arguments"""
```

### 4. Dynamic Entity Creation

Allow for runtime entity configuration:

```python
def create_debater(name, position, style, knowledge_level):
    """Factory function for creating debate entities"""
    # Configure components based on parameters
    # Return customized debater entity
```

## Troubleshooting and Debugging

### Common Issues

1. **Memory Retrieval Issues**: If agents aren't using relevant memories:
   - Check embedding model is working correctly
   - Ensure memory bank is being populated
   - Verify query structure for retrieval

2. **Agent Behavior Problems**: If agents act unexpectedly:
   - Review context components for correctness
   - Check act component implementation
   - Verify context processor logic

3. **Scene Transition Failures**: If scenes aren't progressing:
   - Check scene tracker configuration
   - Verify transition conditions
   - Review game master logic

### Debugging Tools

1. **EntityAgentWithLogging**: Use this class to get detailed logs of agent decision processes
2. **HTML Output**: Review the HTML results for detailed simulation flow
3. **Memory Inspection**: Check memory bank contents during development

## Conclusion

The Concordia framework provides a powerful foundation for creating complex multi-agent simulations. By leveraging its modular architecture, developers can create sophisticated scenarios with intelligent agent interactions.

For further examples, reference the `geo_mod` implementation which demonstrates a complete debate simulation using the Concordia framework.

## References

1. Concordia Source: `concordia/`
2. Geo_Mod Example: `pysrcai/geo_mod/`
3. API Documentation: `concordia/.doc/api_html_autodoc`
