# PySrcAI

A Python-based AI agent framework for simulations, embeddings, and more.

## Overview

PySrcAI is a modular, component-based framework for creating AI agent simulations. It provides:

- **Agent System**: Flexible agent architecture with Actors and Archons
- **Memory System**: Configurable memory components with basic and associative storage
- **Language Models**: Integration with various LLM providers (LM Studio, OpenRouter, etc.)
- **Embedding System**: Vector embeddings and storage for semantic search
- **Environment System**: Configurable simulation environments
- **Component Architecture**: Modular, pluggable components for easy customization

## Project Structure

```
pysrcai/
├── __init__.py                 # Main package API
├── agents/                     # Agent system
│   ├── __init__.py
│   ├── base/                   # Core agent classes
│   │   ├── __init__.py
│   │   ├── agent.py           # Base Agent class
│   │   ├── actor.py           # Actor implementation
│   │   ├── archon.py          # Archon implementation
│   │   └── agent_factory.py   # Agent factory
│   ├── components/             # Agent components
│   │   ├── __init__.py
│   │   ├── component_factory.py
│   │   └── llm_components.py
│   ├── memory/                 # Memory system
│   │   ├── __init__.py
│   │   ├── memory_components.py
│   │   ├── memory_factory.py
│   │   └── embedders.py
│   └── environment/            # Environment components
│       ├── __init__.py
│       └── environment_components.py
├── core/                       # Core simulation engine
│   ├── __init__.py
│   ├── engine.py              # Simulation engine
│   └── factory.py             # Simulation factory
├── llm/                        # Language model system
│   ├── __init__.py
│   ├── language_model.py      # Base LanguageModel class
│   ├── lmstudio_model.py      # LM Studio integration
│   ├── openrouter_model.py    # OpenRouter integration
│   ├── no_language_model.py   # Mock model for testing
│   ├── retry_wrapper.py       # Retry wrapper
│   ├── call_limit_wrapper.py  # Call limit wrapper
│   └── llm_components.py      # LLM agent components
├── embeddings/                 # Embedding system
│   ├── __init__.py
│   ├── base.py                # Base embedder class
│   ├── factory.py             # Embedder factory
│   ├── models/
│   │   ├── __init__.py
│   │   └── sentence_transformers.py
│   └── vectorstore/
│       ├── __init__.py
│       ├── base.py
│       ├── factory.py
│       └── chroma_adapter.py
├── config/                     # Configuration system
│   ├── __init__.py
│   ├── config_loader.py
│   └── embedding_config.py
├── environment/                # Environment system
│   ├── __init__.py
│   └── objects.py
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── concurrency.py
│   ├── measurements.py
│   ├── sampling.py
│   ├── text.py
│   ├── json.py
│   ├── html.py
│   └── plotting.py
├── examples/                   # Examples and demos
│   ├── __init__.py
│   ├── demos/
│   ├── configs/
│   ├── agents/
│   ├── environments/
│   └── scenario/
└── tests/                      # Test suite
    ├── __init__.py
    ├── test_agents/
    ├── test_core/
    ├── test_llm/
    └── test_embeddings/
```

## Quick Start

### Basic Usage

```python
from pysrcai import SimulationFactory, Actor, Archon

# Create a simple simulation
config = {
    'agents': [
        {
            'name': 'Alice',
            'type': 'actor',
            'llm': 'mock',
            'memory': 'basic'
        },
        {
            'name': 'Bob', 
            'type': 'actor',
            'llm': 'mock',
            'memory': 'basic'
        }
    ],
    'engine': {
        'type': 'sequential',
        'steps': 5
    }
}

# Create and run simulation
factory = SimulationFactory()
engine, steps = factory.create_engine(config)
engine.run(steps)
```

### Creating Agents

```python
from pysrcai import Actor, Archon
from pysrcai.llm import LMStudioLanguageModel

# Create an actor with LM Studio
model = LMStudioLanguageModel("my-model")
actor = Actor(
    agent_name="DebateParticipant",
    goals=["Argue for renewable energy"],
    personality_traits={"assertiveness": 0.8}
)

# Create an archon moderator
archon = Archon(
    agent_name="Moderator",
    moderation_rules=["Enforce time limits", "Maintain civility"],
    authority_level="high"
)
```

### Using Memory Components

```python
from pysrcai.agents.memory import BasicMemoryBank, MemoryComponent

# Create memory bank
memory_bank = BasicMemoryBank(max_memories=1000)

# Create memory component
memory_component = MemoryComponent(
    memory_bank=memory_bank,
    max_context_memories=5
)

# Add memories
memory_component.add_explicit_memory(
    "The debate started with Alice arguing for renewable energy",
    tags=["debate", "renewable"],
    importance=0.8
)
```

## Configuration

PySrcAI uses YAML configuration files for defining simulations:

```yaml
# config.yaml
agents:
  - name: Alice
    type: actor
    llm: lmstudio
    memory: basic
    personality:
      assertiveness: 0.8
      knowledge_level: 0.9
    goals:
      - "Argue for renewable energy"
  
  - name: Bob
    type: actor
    llm: openrouter
    memory: associative
    personality:
      assertiveness: 0.6
      knowledge_level: 0.7
    goals:
      - "Argue for nuclear energy"

engine:
  type: sequential
  steps: 10

scenario:
  initial_state:
    topic: "Energy policy debate"
    time_limit: 300
```

## Key Features

### Modular Architecture
- **Component-based**: All functionality is implemented as pluggable components
- **Factory pattern**: Easy creation of agents and components from configuration
- **Extensible**: Add new components without modifying core code

### Memory System
- **Basic Memory**: Simple chronological storage with text search
- **Associative Memory**: Embedding-based semantic search
- **Configurable**: Choose memory type and parameters per agent

### Language Model Integration
- **Multiple Providers**: LM Studio, OpenRouter, mock models
- **Wrappers**: Retry logic, call limiting, error handling
- **Flexible**: Easy to add new LLM providers

### Environment System
- **Configurable**: Define environments in YAML
- **State Management**: Track simulation state
- **Agent Integration**: Connect agents to environment context

## Development

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd pysrcai

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_agents/
python -m pytest tests/test_core/
```

### Adding New Components

1. Create your component class inheriting from the appropriate base class
2. Implement required methods
3. Add to the appropriate factory
4. Update configuration schema if needed
5. Add tests

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Add your license information here] 