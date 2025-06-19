# Concordia Framework Documentation

## Overview

Concordia is a modular framework for building agent-based simulations, focusing particularly on AI agents with complex behaviors. It utilizes a component-based architecture that allows for flexible, extensible agent design through composition rather than inheritance.

## Core Architecture

### Entity System

The foundation of Concordia is the entity system, where each participant in a simulation is represented as an `Entity`. Entities have two fundamental capabilities:
- **Acting**: Determining and executing actions based on their current state and environment
- **Observing**: Processing information from the environment to update their internal state

### Component-Based Design

Concordia implements a component-based architecture where agent functionality is defined through pluggable components rather than through inheritance hierarchies. This approach provides several benefits:
- **Modularity**: Components can be developed, tested, and maintained independently
- **Reusability**: Common behaviors can be encapsulated in components and reused across different agent types
- **Flexibility**: Agent behavior can be modified by adding, removing, or reconfiguring components

## Key Components

### Entity Agent

The `EntityAgent` class is the fundamental building block in Concordia, representing an agent in the simulation:

- **Named Entity**: Each agent has a unique identifier (`agent_name`)
- **Component Management**: Maintains collections of components that define its behavior
  - **Acting Component**: Determines how the agent chooses actions
  - **Context Processor**: Processes information from context components
  - **Context Components**: Provide contextual information for decision-making

### Phase-Based Execution

Agents operate through a series of phases that manage information flow and component interactions:
- **PRE_ACT**: Gathers context before making a decision
- **POST_ACT**: Processes the aftermath of an action
- **PRE_OBSERVE**: Prepares to process new information
- **POST_OBSERVE**: Processes the implications of new information
- **UPDATE**: Commits changes to state
- **READY**: Default state when not in other phases

## Component Types

Concordia defines several key component interfaces:

### Base Component
Foundation for all components. Can be associated with an `EntityWithComponents` and provides state management via `get_state()` and `set_state()`.

### Context Component
Extends `BaseComponent` to provide context during agent lifecycle phases through methods like:
- `pre_act()`
- `post_act()`
- `pre_observe()`
- `post_observe()`
- `update()`

### Acting Component
Specialized component for deciding actions via `get_action_attempt()`.

### Context Processor Component
Processes combined contexts from other components.

## Standard Components

Concordia comes with a variety of pre-built components:

### Memory Management
- **Memory**: Base class for memory management
  - **AssociativeMemory**: Uses a memory bank for storage
  - **ListMemory**: Uses a simple list structure
- **ObservationToMemory**: Adds observations to memory
- **LastNObservations**: Provides access to recent observations

### Agent Cognition
- **Plan**: Manages the agent's planning process
- **QuestionOfRecentMemories**: Asks questions about recent memories
- **SituationPerception**: Analyzes the current situation
- **SelfPerception**: Builds a model of the agent's identity
- **PersonBySituation**: Determines appropriate actions

### Utility Components
- **Constant**: Provides fixed string content
- **Instructions**: Specialized constant for default instructions
- **ReportFunction**: Provides dynamic content from a function
- **ActionSpecIgnored**: For components whose output can be cached

## Prefabricated Entities

Concordia provides ready-to-use entity configurations:

### Basic Entity
Uses the "three key questions" approach:
1. "What situation am I in?"
2. "What kind of person am I?"
3. "What would a person like me do in this situation?"

Components include:
- Instructions
- ObservationToMemory
- LastNObservations
- SituationPerception
- SelfPerception
- PersonBySituation
- AllSimilarMemories

### Basic With Plan
Extends Basic Entity with explicit planning capabilities through a Plan component.

### Minimal Entity
A lean setup with essential components (memory, observations, instructions), configurable with goals and additional components.

## Simulation Flow

### Initialization
1. Choose a prefab and call its `build()` method with required dependencies
2. This creates an `EntityAgentWithLogging` instance with all components initialized

### Observation Cycle
1. Simulation calls `agent.observe(observation_text)`
2. Agent enters `PRE_OBSERVE` phase, calling `pre_observe()` on components
3. Proceeds to `POST_OBSERVE` phase
4. Enters `UPDATE` phase to commit changes
5. Returns to `READY` phase

### Action Cycle
1. Simulation calls `agent.act(action_spec)`
2. Agent enters `PRE_ACT` phase, gathering context from components
3. The context processor integrates information
4. The acting component generates an action attempt
5. Proceeds to `POST_ACT` phase
6. Enters `UPDATE` phase
7. Returns to `READY` and returns the action attempt

## Environment and Simulation

The Environment module provides tools for running simulations:

### Engine
Base class for running simulations, handling:
- Observation generation
- Turn management
- Action processing
- Environment state updates

### Scene Management
Scenes define the structure and rules of specific simulation environments.

## Extending Concordia

New functionality can be added through:
1. **Custom Components**: Create classes implementing the component interfaces
2. **Custom Prefabs**: Combine existing and new components into reusable configurations
3. **Custom Engines**: Implement specialized simulation logic
4. **Custom Language Models**: Connect to different AI backends

## Integration with Language Models

Concordia is designed to work with various language model backends:
-Cloud API: Openrouter via `concordia\language_model\openrouter_model.py`
-Local API: LMStudio via `concordia\language_model\lmstudio_model`

## Development Best Practices

1. **Component Design**: Keep components focused on single responsibilities
2. **Testing**: Use mock language models for predictable testing
3. **Composition**: Build complex behavior through component composition
4. **Documentation**: Document component interfaces and behavior
5. **Phased Execution**: Respect the phase system for predictable behavior

---

This documentation provides a high-level overview of Concordia's architecture and capabilities. 
