# PySrcAI Geopolitical Simulation Framework - Developer Guide

## Quick Start

To run the Phase 2 scene-based debate simulation:

```bash
python pysrcai/geo_mod/simulations/debate.py
```

## Code Flow Analysis: From Start to Simulation End

This section provides a detailed step-by-step analysis of how the geopolitical simulation framework executes, tracing the flow from initialization to completion.

### 1. Entry Point: Simulation Execution (`phase2_debate_fixed.py`)

**File**: `pysrcai/geo_mod/simulations/debate.py`

#### 1.1 Initialization and Setup
```python
# Configure logging for the simulation
configure_logging()

# Initialize OpenRouter language model
model = configure_language_model()

# Set up embedder for agent memory
embedder = configure_embedder()
```

**Flow Details**:
- **Logging Setup**: Configures Python logging to track simulation progress
- **Language Model**: Initializes OpenRouter API connection for LLM interactions
- **Embedder**: Sets up SentenceTransformer for semantic similarity in agent memory

#### 1.2 Simulation Configuration Loading
```python
# Load scenario configuration from russia_ukraine_debate_phase2.py
from pysrcai.geo_mod.scenarios.russia_ukraine_debate import (
    WORD_LIMITS, PREMISE, SCENARIO_SCENES, INSTANCES, GAME_MASTERS
)

# Create simulation configuration object
config = simulation_prefab.Config(
    name="Russia-Ukraine UN Debate",
    premise=PREMISE,
    scenes=SCENARIO_SCENES,
    instances=INSTANCES,
    game_masters=GAME_MASTERS,
)
```

**Flow to**: `pysrcai/geo_mod/scenarios/russia_ukraine_debate_phase2.py`

### 2. Scenario Configuration (`russia_ukraine_debate_phase2.py`)

#### 2.1 Centralized Word Limits
```python
WORD_LIMITS = {
    'opening_statements': {'min': 50, 'max': 100},
    'rebuttals': {'min': 50, 'max': 100},
    'final_arguments': {'min': 100, 'max': 150},
    'judgment': {'min': 100, 'max': 150},
}
```

**Purpose**: Provides centralized configuration for response length constraints across all debate phases.

#### 2.2 Scene Definitions
```python
SCENARIO_SCENES = [
    scene_lib.SceneSpec(
        scene_type=scene_lib.SceneType.CHOICE,
        # Scene configuration with dynamic word limits
    ),
    # ... more scenes
]
```

**Key Components**:
- **SceneType.CHOICE**: Allows entities to make decisions/statements
- **Dynamic Word Limits**: Each scene uses `get_limit_text()` for phase-appropriate constraints
- **Action Specifications**: Define what entities can do in each scene

#### 2.3 Entity Instance Configuration
```python
INSTANCES = [
    prefab_lib.InstanceConfig(
        prefab='nation_entity',
        role=prefab_lib.Role.ENTITY,
        params={
            'name': RUSSIA,
            'goal': "...",
            'context': "...",
            'word_limits': WORD_LIMITS,  # Pass centralized limits
        },
    ),
    # ... more instances
]
```

**Flow to**: Entity prefab creation

### 3. Entity Creation (`nation_entity.py`, `moderator_entity.py`)

#### 3.1 Nation Entity Prefab (`pysrcai/geo_mod/prefabs/entities/nation_entity.py`)

**Key Flow**:
```python
def build(self, model, memory_bank):
    # Extract parameters including word_limits
    word_limits = self.params.get('word_limits')

    # Create dynamic response constraints
    if word_limits:
        response_limit = generate_constraint_text(word_limits)
    else:
        response_limit = default_constraint_text()

    # Build agent with memory, observation, and decision components
    return EntityAgentWithLogging(...)
```

**Components Created**:
- **Memory System**: Uses FormativeMemoryFactory with embeddings
- **Observation Components**: Tracks recent events and statements
- **Decision Components**: Handles goal-oriented behavior
- **Context Components**: Maintains identity and relationships

#### 3.2 Moderator Entity Prefab (`pysrcai/geo_mod/prefabs/entities/moderator_entity.py`)

**Purpose**: Creates neutral moderator with assessment capabilities
- **Neutrality**: No national bias or goals
- **Assessment**: Can evaluate arguments and declare winners
- **Process Management**: Guides debate flow and maintains order

### 4. Concordia Framework Integration

#### 4.1 Simulation Object Creation (`concordia/prefabs/simulation.py`)

**Flow**:
```python
simulation = simulation_prefab.Simulation(
    config=config,
    model=model,
    memory_bank_factory=memory_bank_factory,
)
```

**Concordia Components Used**:
- **Simulation Prefab**: High-level simulation orchestrator
- **Memory Bank Factory**: Creates isolated memory for each entity
- **Scene Management**: Handles scene transitions and constraints

#### 4.2 Game Master Selection (`concordia/prefabs/game_master/dialogic_and_dramaturgic.py`)

**Key Flow**:
```python
# Concordia automatically selects D&D GameMaster based on config
game_master = DialogicAndDramaturgicGameMaster(
    model=model,
    memory_bank=memory_bank,
    clock=clock,
    # Built-in components
    components=[
        scene_tracker.SceneTracker(),  # Automatic scene management
        turn_taking.TurnTaking(),      # Entity turn coordination
        # ... other components
    ]
)
```

**Concordia Source**: `concordia/prefabs/game_master/dialogic_and_dramaturgic.py`

#### 4.3 Scene Tracker Component (`concordia/components/game_master/scene_tracker.py`)

**Automatic Scene Management**:
- **Scene Progression**: Automatically moves between debate phases
- **Constraint Enforcement**: Applies scene-specific rules and limits
- **Turn Management**: Ensures proper entity participation order

### 5. Simulation Execution Loop

#### 5.1 Core Execution (`concordia/environment/engine.py`)

**Flow**:
```python
while not terminate_simulation:
    # 1. Game Master determines next actor
    next_entity = game_master.next_entity()

    # 2. Entity observes current state
    observation = entity.observe()

    # 3. Entity makes decision based on current scene
    action = entity.act(observation)

    # 4. Action is processed and resolved
    event = resolve_action(action)

    # 5. All entities observe the resolved event
    broadcast_event(event)

    # 6. Check termination conditions
    terminate = check_termination_conditions()
```

#### 5.2 Entity Decision Making (`concordia/agents/entity_agent_with_logging.py`)

**Decision Process**:
1. **Observation**: Entity receives current scene context
2. **Memory Retrieval**: Relevant memories accessed via embeddings
3. **Context Assembly**: Current goals, constraints, and recent events
4. **LLM Generation**: Language model generates appropriate response
5. **Action Formatting**: Response formatted according to scene requirements

#### 5.3 Scene Transitions

**Automatic Progression**:
- **Scene Tracker**: Monitors scene completion conditions
- **D&D GameMaster**: Coordinates scene transitions
- **Entity Notification**: All entities receive scene change updates
- **Constraint Updates**: New scene limits automatically applied

### 6. Simulation Completion and Output

#### 6.1 Termination Detection
- **Scene Completion**: All required scenes completed
- **Winner Declaration**: Moderator provides final judgment
- **Simulation State**: No more actions required

#### 6.2 Results Generation
```python
# Generate HTML report with full conversation history
generate_html_results("phase2_debate_results.html")

# Log completion status
logger.info("Scene-based simulation completed successfully!")
```

## Implemented Features Summary

### ✅ Core Features Implemented

1. **Scene-Based Debate Structure**
   - 4-phase debate: Opening → Rebuttals → Final Arguments → Judgment
   - Automatic scene progression via Concordia's SceneTracker
   - Phase-specific constraints and prompting

2. **Centralized Word Limit System**
   - Configurable word limits per debate phase
   - Dynamic constraint generation for entities
   - Consistent enforcement across all scenes

3. **Specialized Entity Types**
   - **Nation Entities**: Goal-oriented diplomatic representatives
   - **Moderator Entity**: Neutral debate facilitator with assessment capabilities
   - **Dynamic Configuration**: Entities adapt to provided parameters

4. **Advanced Memory System**
   - Semantic embeddings for memory retrieval
   - Formative memories for identity consistency
   - Context-aware decision making

5. **Professional Orchestration**
   - Dialogic & Dramaturgic GameMaster integration
   - Turn-based coordination between entities
   - Diplomatic protocol enforcement

6. **Comprehensive Logging & Output**
   - Detailed simulation logging
   - HTML report generation
   - Progress tracking throughout execution

### 🔄 Concordia Framework Features Utilized

#### From `concordia/prefabs/`
- **Simulation Prefab**: High-level simulation orchestration
- **D&D GameMaster**: Scene-based debate management
- **Entity Agent**: Core agent behavior and decision making

#### From `concordia/components/`
- **SceneTracker**: Automatic scene progression
- **AssociativeMemory**: Semantic memory retrieval
- **Observation Components**: Event tracking and context awareness
- **TurnTaking**: Entity coordination and ordering

#### From `concordia/typing/`
- **Scene Specifications**: Structured scene definitions
- **Entity Components**: Modular entity behavior
- **Prefab System**: Reusable component architecture

### 🚧 Potential Future Enhancements

#### Not Yet Implemented from Concordia

1. **Advanced Clock Systems** (`concordia/clocks/`)
   - **GameClock**: Time-based simulation progression
   - **Scheduled Events**: Time-triggered debate phases
   - **Temporal Constraints**: Real-time debate limits

2. **Multi-GameMaster Scenarios** (`concordia/prefabs/game_master/`)
   - **Partisan GameMaster**: Biased perspective management
   - **MultipleGameMasters**: Competing moderation approaches
   - **GameMaster Networks**: Collaborative orchestration

3. **Advanced Entity Networks** (`concordia/components/agent/`)
   - **Relationship Components**: Dynamic entity relationships
   - **Coalition Formation**: Alliance building during debates
   - **Reputation Systems**: Track diplomatic standing

4. **Enhanced Memory Systems** (`concordia/associative_memory/`)
   - **Importance Scoring**: Priority-based memory retrieval
   - **Memory Decay**: Time-based memory fading
   - **Shared Memories**: Common knowledge between entities

5. **Complex Environment Features** (`concordia/environment/`)
   - **Multi-Environment**: Parallel debate scenarios
   - **Environment State**: Persistent world conditions
   - **External Events**: Real-world event integration

6. **Advanced Evaluation** (`concordia/utils/`)
   - **Argument Scoring**: Automated debate assessment
   - **Consensus Tracking**: Agreement measurement
   - **Performance Metrics**: Quantitative evaluation

### 📈 Recommended Next Steps

1. **Scenario Expansion**: Create additional debate topics using the modular framework
2. **Evaluation Integration**: Implement automated argument scoring
3. **Multi-Round Debates**: Extend to tournament-style competitions
4. **Real-Time Integration**: Connect to live geopolitical data feeds
5. **Visualization**: Add interactive debate flow visualization
6. **Performance Optimization**: Enhance response generation speed

## Architecture Benefits

The current implementation provides:

- **Modularity**: Easy to create new debate scenarios
- **Scalability**: Framework supports additional entities and complexity
- **Maintainability**: Clean separation between configuration and logic
- **Extensibility**: Ready for Concordia framework enhancements
- **Reliability**: Robust error handling and state management