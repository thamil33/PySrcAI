# Project Development Overview

## The goal is to develop a scalable and modular framework within `pyscrai` that integrates seamlessly with the `concordia` framework. This approach emphasizes building upon foundational components in a flexible and extensible manner.

# Current Project Status

## Component Overview

### Embedding System
- SentenceTransformer for text embeddings
- Model: "sentence-transformers/all-mpnet-base-v2"
- CPU and CUDA support

### Language Model Integration
- OpenRouter API with environment-based API key management

### Memory System
- AssociativeMemoryBank with runtime-only storage
- Components: LastNObservations (100 entries), AssociativeMemory

### Debate Engine
- Turn-based debate management
- Event history tracking and participant management

### Game Master
- Observation and memory components
- Event logging capabilities

### Entity System
- Two entities (Demon and Angel) with defined personalities and goals
- Shared memory bank

## Current Functionality

### Core Features
- Philosophical debate simulation
- Turn-based entity interaction
- Memory-aware and context-driven responses

### Limitations
- No persistent storage
- Basic debate structure enforcement
- Limited analysis or metrics collection


# Project Directory Overview

## concordia/
Contains core modules for the Concordia framework, including agents, memory, clocks, and components.

### agents/
Defines entity agents and their logging mechanisms.

### associative_memory/
Implements basic and formative associative memory structures.

### clocks/
Manages game clock functionality and tests.

### components/
Includes modular components for agents, game masters, and deprecated features.

### contrib/
Houses contributed components and deprecated modules.

### document/
Handles document-related operations, including interactive documents and tests.

### environment/
Manages the simulation environment, including engines and scenes.

### language_model/
Provides wrappers and implementations for various language models (e.g., Ollama, OpenRouter, Mistral).

### prefabs/
Contains pre-configured templates for entities, game masters, and simulations.

### testing/
Includes mock models and testing utilities.

### thought_chains/
Implements thought chain logic and deprecated features.

### typing/
Defines type annotations for entities, components, scenes, and simulations.

### utils/
Offers utility functions for concurrency, helper methods, measurements, and plotting.

## docs/
Contains documentation for the project, including development plans, features, and observability guides.

## pyscrai/
Core package for the project, including components, embedding, engine, and scenario modules.


### components/
Contains modular components for the `pyscrai` package.

### embedding/
Handles embedding-related functionalities.

### engine/
Manages the engine logic for simulations.

### scenario/
Defines scenarios for the `pyscrai` package.