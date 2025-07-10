# Enhanced Modular Demo

## Overview

This is a demonstration of the enhanced modular framework for PySrcAI. It showcases:

- The module system for pluggable components
- Event-driven architecture for cross-component communication
- State management with history tracking
- Environment abstraction
- Debate as a pluggable module

## Architecture

The enhanced framework is built around these core components:

1. **Module System** - Provides registration, discovery, and dependency management for modules
2. **Event System** - Implements a publish/subscribe pattern for cross-module communication
3. **State Management** - Generic state container with history tracking and serialization
4. **Enhanced Engine** - Core simulation engine with module support
5. **Environment Interface** - Common interface for all environment modules

## Key Files

- `pysrcai/core/module_system.py` - Core module registration and discovery
- `pysrcai/core/enhanced_engine.py` - Enhanced simulation engine
- `pysrcai/core/event_system.py` - Event bus implementation
- `pysrcai/core/state_management.py` - State management utilities
- `pysrcai/agents/environment/module_interface.py` - Environment module interfaces
- `pysrcai/modules/debate/debate_module.py` - Example debate module
- `pysrcai/agents/environment/enhanced_minimal.py` - Enhanced minimal environment
- `pysrcai/demos/enhanced_demo.py` - Demo script

## Running the Demo

To run the enhanced demo:

```bash
python run_enhanced_demo.py
```

Or use the module directly:

```bash
python -m pysrcai.demos.enhanced_demo
```

## Configuration

The demo uses a configuration file located at `pysrcai/config/prefabs/enhanced_minimal.yaml`. 
You can customize this file to:

- Change the debate topic
- Adjust the number of steps
- Configure the agents and archon
- Set the language model (mock, lmstudio, or openrouter)

## Extending the Framework

To add a new module:

1. Create a new module class that extends the appropriate interface
2. Use the `@register_module` decorator to register it
3. Implement the required methods
4. Add configuration in the YAML file

Example:
```python
from pysrcai.core.module_system import register_module, ModuleInterface, ModuleType

@register_module("my_module", version="0.1.0", dependencies=["core"])
class MyModule(ModuleInterface):
    @property
    def module_type(self) -> ModuleType:
        return ModuleType.ENGINE
        
    def initialize(self, config):
        # Your initialization code here
        pass
```

## Event System Usage

To use the event system:

```python
from pysrcai.core.event_system import event_bus

# Subscribe to events
event_bus.subscribe("my_event_type", my_handler_function)

# Publish events
event_bus.publish_event("my_event_type", {"key": "value"})
```

## State Management

To use the state container:

```python
from pysrcai.core.state_management import JsonSerializableStateContainer

# Create a state container
state = JsonSerializableStateContainer({"counter": 0})

# Update state
state.update(lambda s: {**s, "counter": s["counter"] + 1})

# Save state
state.save_to_file("state.json")
```
