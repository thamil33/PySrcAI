# Project Instructions for ChatGPT Codex

## Project Overview

This is a private development repository containing the **pyscrai** package and the **Concordia** framework for building modular agent-based simulations. The project focuses on creating extensible, robust AI agent simulations using a component-based architecture.

## Repository Structure

```
pyscrai_workstation/
├── pyscrai/                    # Core pyscrai package
│   ├── components/             # Simulation components
│   ├── embedding/              # Text embedding functionality
│   ├── engine/                 # Simulation engines
│   └── scenario/               # Scenario definitions
├── concordia/                  # Concordia framework
│   ├── .api_docs/              # Auto-generated API documentation
│   ├── agents/                 # Agent implementations
│   ├── components/             # Framework components
│   ├── language_model/         # LLM integrations
│   ├── prefabs/                # Pre-built entities
│   └── typing/                 # Type definitions
├── docs/                       # Documentation
│   ├── concordia/              # Framework documentation
│   ├── concordia_developers_guide.md
│   └── dev_plan.md             # Current development status
└── requirements.txt            # Python dependencies
```

## Current Implementation Status

- **Framework**: Concordia is fully integrated and functional
- **LLM Integration**: OpenRouter API via `concordia/language_model/openrouter_model.py`
- **Text Embeddings**: 
  - **Primary**: HuggingFace Inference API (free, cloud-based) via `pyscrai/embedding/hf_embedding.py`
  - **Alternative**: Local SentenceTransformers (optional, for privacy/offline use)
- **Configuration**: Environment variables in `.env` file
- **Testing**: Integration tests validated and documented
- **Dependencies**: Modular installation with optional PyTorch for local embeddings

## Development Guidelines

### Branch Management
- **Primary Branch**: Use the `codex` branch for all development
- **Repository Type**: Private, internal developer use only
- **Security**: No licensing or API exposure concerns required

### Architecture Principles
- **Component-Based Design**: Use Concordia's modular component system
- **Extensibility**: Build for easy extension and modification
- **Documentation**: Maintain comprehensive documentation alongside development
- **Testing**: Implement robust testing patterns

### Key Documentation References
1. **[Concordia Overview](docs/concordia/concordia_overview.md)** - Framework architecture
2. **[Developer's Guide](docs/concordia_developers_guide.md)** - Comprehensive development guide
3. **[API Documentation](concordia/.api_docs/README.md)** - Auto-generated API reference (LLM-compatible JSON format)
4. **[Integration Tests](concordia/concordia_integration_test.md)** - Validation and examples
5. **[Development Plan](docs/dev_plan.md)** - Current status and roadmap

## Development Workflow

1. **Setup**: Follow `readme.md` for environment setup
2. **Development**: Work within the `pyscrai` directory for new features
3. **Testing**: Use integration tests to validate functionality
4. **Documentation**: Update relevant docs with each change
5. **Status**: Keep `docs/dev_plan.md` current with progress

## Current Focus Areas

- Expanding pyscrai module functionality
- Enhancing component library
- Improving simulation capabilities
- Maintaining comprehensive documentation

> **Always refer to `docs/dev_plan.md` for the most current development status and immediate priorities.**



