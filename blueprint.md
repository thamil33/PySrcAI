# PySrcAI Blueprint

## Project Overview

PySrcAI is a modular multi-agent simulation framework designed to create sophisticated, scalable simulations with AI-powered entities. Built as an evolution of the Concordia framework, PySrcAI addresses the limitations of monolithic simulation architectures by introducing true modularity, cleaner abstractions, and enhanced flexibility.

### Core Vision

PySrcAI enables developers to build complex multi-agent simulations where AI entities can interact, debate, collaborate, and compete in various scenarios. The framework is designed to support everything from debate engines and negotiation simulations to complex social dynamics and strategic games.

## Architectural Philosophy

### From Concordia to PySrcAI

PySrcAI was born from experience with the Concordia framework and lessons learned from the `geo_mod` implementation. While Concordia provided powerful tools for multi-agent simulations, it suffered from:

- **Monolithic structure** that made modular development difficult
- **Unclear entity hierarchies** with confusing terminology
- **Tight coupling** between components that hindered reusability
- **Limited scalability** for complex, multi-domain simulations

PySrcAI addresses these issues through:

- **True modularity** with shared components and domain-specific modules
- **Clear entity abstractions** with logical hierarchies
- **Loose coupling** through well-defined interfaces
- **Horizontal scalability** supporting multiple simulation types

### Core Principles

1. **Modularity First**: Every component should be reusable across different simulation types
2. **Clear Abstractions**: Entity roles and responsibilities should be immediately obvious
3. **Configuration-Driven**: Simulations should be configurable without code changes
4. **AI-Native**: Built from the ground up for AI-powered entities
5. **Developer-Friendly**: Easy to understand, extend, and maintain

## Architecture Overview

```
pysrcai/
├── src/                          # Shared components and infrastructure
│   ├── agents/                   # Complete agent system
│   │   ├── agent.py             # Base Agent class with component system
│   │   ├── actor.py             # Simulation participants
│   │   ├── archon.py            # Moderators and orchestrators  
│   │   ├── llm_components.py    # LLM integration with role-specific prompting
│   │   ├── memory/              # Memory system with multiple strategies
│   │   │   ├── memory_components.py
│   │   │   └── embedders.py
│   │   └── demo/                # Working demonstrations
│   │       ├── demo_agents.py
│   │       ├── demo_llm_agents.py
│   │       └── demo_memory.py
│   ├── components/               # Reusable building blocks
│   ├── engines/                  # Core simulation architecture  
│   ├── factories/                # Configuration and building tools
│   ├── language_model_client/    # LLM integration (from Concordia)
│   ├── utils/                    # Shared utilities
├── mod_debate/                   # Debate simulation module
├── mod_scenario/                 # General scenario module
└── mod_strategy/                 # Strategy game module
```

## Entity Hierarchy

PySrcAI introduces a clear, three-tier entity hierarchy:

### Entity (Abstract Base)
The highest-level abstraction representing any autonomous agent in the simulation.

### Agents (Specialized Entities)
Two primary types of agents with distinct roles:

#### Actors
- **Purpose**: Direct simulation participants
- **Examples**: Debate participants, game players, negotiators
- **Instantiation**: Through module-specific configurations
- **Characteristics**: Goal-oriented, competitive/collaborative, specialized behaviors

#### Archons
- **Purpose**: Simulation moderators and orchestrators  
- **Examples**: Debate moderators, game masters, environment controllers
- **Instantiation**: Usually through `src/` infrastructure, customized by modules
- **Characteristics**: Administrative, rule-enforcing, environment-managing

### Key Differences from Concordia

| Concordia | PySrcAI | Improvement |
|-----------|---------|-------------|
| "entities" | Actors | Clear participant role |
| "game_masters" | Archons | Broader, more flexible moderator concept |
| Mixed responsibilities | Separated concerns | Better maintainability |

## Component Architecture

### `/src/agents/`
Contains the complete agent system with base classes and shared components:

#### Core Agent Classes
- **`agent.py`** - Base Agent class with component system, state management, and thread safety
- **`actor.py`** - Specialized agents for simulation participants (players, debaters, negotiators)
- **`archon.py`** - Specialized agents for moderators and orchestrators (game masters, environment controllers)

#### LLM Integration
- **`llm_components.py`** - LLM-powered acting components with role-specific prompt engineering
- Support for both Actor and Archon LLM interactions with appropriate context

#### Memory System
- **`memory/`** - Complete memory subsystem:
  - `memory_components.py` - BasicMemoryBank, AssociativeMemoryBank, MemoryComponent
  - `embedders.py` - Simple embedding functions for memory retrieval
- Memory integration with agent context system
- Multiple retrieval strategies (recent, query-based, tag-based)
- Thread-safe memory operations

#### Demonstrations
- **`demo/`** - Working examples showcasing system capabilities:
  - `demo_agents.py` - Basic agent functionality
  - `demo_llm_agents.py` - LLM-powered intelligent agents
  - `demo_memory.py` - Memory integration with persistent, intelligent behavior

### `/src/components/`
Organized by application type:
- `actor/` - Components specific to simulation participants
- `archon/` - Components specific to 'game masters' / Archons
- `data/` - References and Data files.

### `/src/engines/`
Core simulation infrastructure:
- Execution engines
- Event systems
- State management
- `/types/` - Base and abstract classes (replaces Concordia's `/types`)

### `/src/factories/engines`
Configuration and instantiation tools:
- Agent builders using prefabs
- Scenario configurators
- Template systems
- Module integration tools

## Module System

Modules (`mod_*`) are domain-specific implementations that leverage the shared `src/` infrastructure:

### Current Modules

#### `mod_debate/` - Debate Engine
A sophisticated debate simulation system featuring:
- Multi-participant debates
- Various debate formats (formal, informal, structured)
- Real-time moderation and scoring
- Argument analysis and tracking

#### `mod_scenario/` - General Scenarios
Flexible framework for creating custom simulation scenarios without specialized domain logic.

#### `mod_strategy/` - Strategy Games
Framework for competitive strategy simulations including games, negotiations, and competitive scenarios.

### Module Structure
Each module typically contains:
```
mod_example/
├── session.py           # Main session orchestrator
├── agents/             # Module-specific agent implementations
├── environment/        # Domain-specific environment logic
├── scenarios/          # Pre-configured scenario templates
└── templates/          # Reusable configuration templates
```

## Development Philosophy

### Modular Development
- **Shared First**: Build reusable components in `src/` before module-specific implementations
- **Interface-Driven**: Define clear contracts between components
- **Configuration Over Code**: Prefer configuration-based customization using config / yaml files. Easy to port to an API and abstract a layer for UI. 

### Migration Strategy
Use `Concordia` directory to do the following: 
- Identify reusable components
- Adapt to PySrcAI's architecture
- Integrate with the new entity hierarchy
- Maintain backward compatibility where beneficial

## Language Model Integration

PySrcAI features enhanced language model integration with production-ready capabilities:

### Supported Platforms
- **Local LLM** via LM Studio with automatic fallback and error handling
- **Cloud LLM** via OpenRouter with comprehensive model support
- **Mock Models** for testing and development

### Key Features
- **Role-Specific Prompting**: Actors and Archons receive contextually appropriate prompts
- **Memory-Driven Context**: LLMs receive relevant memory context for intelligent decision-making
- **Robust Error Handling**: Automatic retry logic and graceful degradation
- **Thread Safety**: Concurrent LLM access with proper synchronization
- **Configurable Model Selection**: Per-agent model customization
- **Built-in Rate Limiting**: Respectful API usage with retry and backoff

### Integration Architecture
- **ActorLLMComponent**: Participant-focused prompting and behavior
- **ArchonLLMComponent**: Administrative and analytical prompting
- **MemoryComponent**: Provides memory context to LLM prompts
- **Language Model Client**: Unified interface supporting multiple providers

## Recent Development Achievements

PySrcAI has reached a major milestone with the completion of its foundational architecture and successful integration of intelligent, memory-driven agents.

### Foundation Complete ✅
The core PySrcAI infrastructure is now production-ready:
- **Agent Hierarchy**: Clear Actor/Archon distinction with specialized roles
- **Component System**: Modular, reusable components with clean interfaces  
- **LLM Integration**: Real AI-powered decision making with role-specific prompting
- **Memory System**: Persistent, intelligent memory with multiple retrieval strategies
- **Thread Safety**: Concurrent operations with proper synchronization
- **Robust Error Handling**: Production-ready reliability and graceful degradation

### Demonstrated Capabilities ✅
Working demonstrations showcase sophisticated AI behavior:
- **Memory-Driven Interactions**: Agents remember previous conversations and context
- **Intelligent Recognition**: Agents recognize and build upon past relationships
- **Contextual Decision Making**: LLMs receive relevant memory context for informed responses
- **Persistent Relationships**: Memory enables ongoing agent relationships across sessions
- **Role-Appropriate Behavior**: Actors and Archons exhibit contextually appropriate behavior

### Real AI Integration ✅
Successfully tested with production language models:
- **OpenRouter Integration**: Cloud-based LLM access with comprehensive model support
- **LM Studio Support**: Local LLM deployment for privacy and control
- **Intelligent Responses**: Agents demonstrate contextual awareness and memory-driven behavior
- **Sophisticated Memory Recall**: Agents reference specific past interactions with detailed accuracy

### Next Phase Ready ✅
The foundation enables rapid development of domain-specific modules:
- **Debate Engine**: Ready for sophisticated multi-participant debate simulations
- **Strategy Games**: Framework for competitive and collaborative scenarios
- **Social Simulations**: Support for complex relationship and community dynamics
- **Educational Tools**: Interactive learning environments with persistent AI tutors

This milestone represents the successful evolution from Concordia's limitations to a truly modular, scalable, and intelligent multi-agent simulation framework.

## Current Development Status

### Completed
- ✅ Core architecture design
- ✅ Directory structure and organization
- ✅ Entity hierarchy definition (Actor/Archon)
- ✅ Language model client (ported from Concordia)
- ✅ Base Agent class with component system
- ✅ Actor and Archon specialized agent classes
- ✅ LLM integration with role-specific components
- ✅ Complete memory system with multiple strategies
- ✅ Memory-based context components
- ✅ Thread-safe agent state management
- ✅ Real AI testing with OpenRouter/LMStudio
- ✅ Demonstration scripts with intelligent behavior

### In Progress
- 🔄 Core engine architecture
- 🔄 Factory pattern implementation  
- 🔄 Debate engine development

### Planned
- ⏳ Component library expansion
- ⏳ YAML-based configuration system
- ⏳ Scenario template system
- ⏳ Advanced documentation and examples
- ⏳ Comprehensive testing framework
- ⏳ Performance optimization
- ⏳ Multi-simulation orchestration

## Getting Started

### For New Developers

1. **Understand the Entity Hierarchy**: Start with the Actor/Archon distinction
2. **Explore the Complete Source Structure**: Examine the `src/agents/` implementation
3. **Run the Demonstrations**: Execute `demo_memory.py` to see intelligent agents in action
4. **Study Working Examples**: The demo scripts showcase memory integration and LLM capabilities
5. **Review Concordia Documentation**: Located at `concordia/.doc/` for background context
6. **Check Working Integration**: Examine `geo_mod/` for integration patterns (though now superseded by modular approach)

### Testing and Validation

PySrcAI has been thoroughly tested with real AI systems:

#### Memory Integration Testing ✅
- **Persistent Context**: Agents successfully store and retrieve interaction history
- **Intelligent Recall**: Memory-driven responses show contextual awareness
- **Multiple Strategies**: Recent, query-based, and tag-based memory retrieval all functional
- **Cross-Session Persistence**: Memory maintains state across agent interactions

#### LLM Integration Validation ✅  
- **OpenRouter Success**: Production testing with cloud-based models (mistralai/mistral-small variants)
- **LM Studio Support**: Local model integration with automatic fallback
- **Role-Specific Prompting**: Actors and Archons receive appropriate context and instructions
- **Error Handling**: Robust retry logic and graceful degradation under various failure scenarios

#### Agent Behavior Verification ✅
- **Intelligent Interactions**: Agents demonstrate sophisticated, memory-driven conversations
- **Relationship Building**: Persistent memory enables ongoing agent relationships
- **Contextual Decision Making**: LLMs use memory context for informed responses
- **Technical Discussions**: Agents maintain detailed technical conversations across sessions

### For AI Assistants

When working with PySrcAI:
1. **Understand the Foundation**: The core agent system with Actor/Archon distinction is complete and production-ready
2. **Respect the Modularity**: Always consider whether code belongs in `src/` (shared infrastructure) or a specific module
3. **Follow Entity Conventions**: Use Actor/Archon terminology consistently
4. **Leverage Existing Components**: The `src/` foundation includes agents, memory, LLM integration, and utilities
5. **Build on Success**: Memory integration, LLM components, and intelligent behavior are fully functional
6. **Document Patterns**: This is a learning system - document architectural decisions
7. **Ready for Modules**: The foundation supports building domain-specific modules like debate engines

## Future Vision

PySrcAI aims to become the de facto framework for AI-powered multi-agent simulations, supporting:

- **Research Applications**: Social dynamics, behavioral studies, AI alignment research
- **Educational Tools**: Interactive learning environments, debate training, strategic thinking
- **Commercial Applications**: Customer service simulations, market modeling, negotiation training
- **Entertainment**: AI-powered games, interactive storytelling, competitive AI contests

The modular architecture ensures that PySrcAI can grow to support these diverse applications while maintaining clean, maintainable code and clear development patterns.

## Contributing

When contributing to PySrcAI:

1. **Start with Issues**: Check existing issues or create new ones for discussion
2. **Follow Architecture**: Respect the src/mod separation and entity hierarchy
3. **Write Tests**: Include tests for new components and modules
4. **Document Changes**: Update this blueprint and relevant documentation
5. **Consider Reusability**: Always ask "Could this be useful in other modules?"

---

*This blueprint is a living document that evolves with the project. As new patterns emerge and the architecture matures, this document should be updated to reflect the current state and future direction of PySrcAI.*