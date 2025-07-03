# Environment Objects & Interactivity Implementation Guide

This guide explains the environment objects system added to PySrcAI, which allows agents to interact with a rich, dynamic world.

## Overview

The environment objects system introduces:

1. **Environment Objects**: Interactive entities in the simulation that agents can perceive and manipulate
2. **Locations**: Containers for objects that represent different areas
3. **Items**: Objects that can be collected, used, or otherwise interacted with
4. **Enhanced Engine**: A simulation engine that processes agent interactions with the environment
5. **YAML Configuration**: An expanded schema for defining environments

## Key Components

### Environment Classes

- `EnvironmentObject`: Base class for all objects in the environment
- `Location`: A place that can contain objects and be visited by agents
- `Item`: An object that can be interacted with (picked up, used, read, etc.)
- `Environment`: Manager class that tracks all locations, items, and state

### Enhanced Engine

The `EnhancedEngine` extends the `SequentialEngine` to:
- Process natural language actions from agents
- Update the environment based on agent actions
- Provide feedback to agents about their interactions
- Maintain an inventory system for agents
- Support the Moderator/Archon in guiding the scenario

## Interaction Model

1. **Agent Action**: Agent uses natural language to describe their action
2. **Action Parsing**: Engine interprets the action using keyword matching
3. **Environment Update**: Relevant objects are updated based on the action
4. **Feedback**: Agent receives a description of the action's outcome
5. **State Tracking**: Scenario state is updated to reflect changes

## Supported Interactions

The system currently supports these basic interactions:

- **Examine/Look**: View descriptions of objects, locations
- **Search**: Look inside containers to find hidden items
- **Take/Pick up**: Add portable items to an agent's inventory
- **Use**: Use items, potentially on other objects
- **Read**: Read content from readable objects

## YAML Configuration

Environment objects are defined in the scenario configuration under `initial_state.environment`:

```yaml
environment:
  locations:
    room_id:
      name: "Room Name"
      description: "Description of the room."
      objects:
        object_id:
          name: "Object Name"
          description: "Description of the object."
          properties:
            # Object-specific properties
            searchable: true
            contents: ["item_id1", "item_id2"]
  items:
    item_id:
      name: "Item Name"
      description: "Description of the item."
      properties:
        # Item-specific properties
        portable: true
        usable: true
        readable: true
        content: "Text content for readable items"
```

## Object Properties

Objects can have various properties that determine their behavior:

- **searchable**: Can be searched to find contained items
- **locked**: Requires a key or action to unlock
- **contents**: List of item IDs contained within
- **portable**: Can be picked up and carried
- **usable**: Can be used by agents
- **usable_on**: List of object IDs this item can be used on
- **readable**: Contains text that can be read
- **content**: The text content of readable objects

## How to Use

### 1. Define Environment in YAML

Create or modify a scenario YAML file to include environment objects:

```yaml
scenario:
  description: "A scenario with interactive objects."
  initial_state:
    environment:
      # Define locations and items here
```

### 2. Use the Enhanced Engine

```python
from pysrcai.src.config.config_loader import load_config
from pysrcai.src.environment.enhanced_factory import EnhancedSimulationFactory

# Load config
config = load_config("path/to/config.yaml")

# Create engine with environment
factory = EnhancedSimulationFactory()
engine, steps = factory.create_engine(config)

# Run the simulation
engine.run(steps=steps)
```

## Example Interactions

Here are examples of how agents might interact with the environment:

- "I look around the room"
- "I examine the desk"
- "I search the bookshelf"
- "I take the key"
- "I use the key on the locked door"
- "I read the note"

## Extending the System

### Adding New Interaction Types

To add new types of interactions:
1. Add detection in the `process_agent_action` method
2. Implement handling in `Environment.process_interaction`
3. Update relevant objects with new properties

### Adding Special Effects

For special interactions or effects:
1. Add custom logic in `process_agent_action`
2. Update the scenario state with new information
3. Optionally, use the Archon to describe special effects

## Best Practices

1. **Clear Descriptions**: Write clear, detailed descriptions for objects
2. **Consistent IDs**: Use consistent, lowercase IDs for objects and items
3. **Reasonable Properties**: Only add properties that make sense for each object
4. **Progressive Complexity**: Start with simple interactions and build up
5. **Archon Guidance**: Use the Archon to help guide agents when they're stuck

## Next Steps

Potential enhancements to the environment system:

1. More sophisticated NLP parsing of agent actions
2. Support for agent movement between multiple locations
3. Time-based events or changes to the environment
4. More complex object interactions and relationships
5. Dynamic environment changes based on agent actions
