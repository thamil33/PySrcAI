# Current Project State

## Project Structure
```
pyscrai/
├── embedding/
│   └── sentence_transformers.py     # Sentence transformer implementation for embeddings
├── engine/
│   ├── debate_engine.py            # Core debate simulation engine
│   └── debate_master.py            # Debate moderator implementation
└── scenario/
    └── angeldemon.py               # Example philosophical debate scenario
```

## Component Overview

### Embedding System
- Using SentenceTransformer for text embeddings
- Model: "sentence-transformers/all-mpnet-base-v2"
- Configured for both CPU and CUDA support
- Error handling for embedding generation

### Language Model Integration
- Using OpenRouter API
- Current model: mistralai/devstral-small:free
- API key management through environment variables
- Direct integration with Concordia's language model interface

### Memory System
- Basic AssociativeMemoryBank implementation
- Runtime-only memory storage
- Memory components:
  - LastNObservations (100 entry history)
  - AssociativeMemory for context storage

### Debate Engine
- Turn-based debate management
- Basic termination conditions (turn count)
- Event history tracking
- Participant management
- Action specification generation

### Game Master (Debate Moderator)
- Basic observation and memory components
- SwitchAct component for entity management
- Passive observation role
- Event logging capabilities

### Entity System
- Two primary entities (Demon and Angel)
- Defined personalities and goals
- Built using BasicEntity prefab
- Shared memory bank between entities

## Current Functionality

### Core Features
1. Basic philosophical debate simulation
2. Turn-based interaction between entities
3. Memory-aware responses
4. Contextual awareness in conversations
5. Basic debate flow management

### Entity Interaction
- Alternating turns between participants
- Context-aware responses
- Personality-driven arguments
- Goal-oriented debate participation

### Memory and Context
- Short-term conversation history
- Basic associative memory
- Observation system for tracking events
- Runtime context maintenance

### Limitations
1. No persistent storage
2. Basic game master functionality
3. Simple termination conditions
4. Limited debate structure enforcement
5. No analysis or metrics collection

## Technical Integration
- Successfully integrated with Concordia framework
- Environment-based configuration
- WSL2 compatibility
- Custom component implementations
