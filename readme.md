# PyScrAI

## 🎯 Project Overview

**`PyScrAI` is a comprehensive framework engineered for broad multi-domain applications while maintaining exceptional quality, performance, and modularity. The platform enables seamless configuration for highly controlled, specialized use cases across diverse sectors and industries.**

**`PyScrAI` delivers advanced tooling for creating and deploying sophisticated agent-based scenarios, with particular emphasis on simulations and cognitive reasoning AI applications. The platform integrates multiple leading frameworks and technologies to provide a unified development environment for building enterprise-grade multi-agent systems, simulations, and AI solutions.**


### 🔬 Target Application Domains

- **Academic Research**: Political science, international relations, and AI safety research
- **Policy Analysis**: Strategic scenario planning for government agencies and NGOs  
- **AI Development**: Multi-agent system research, development, and deployment
- **Educational**: Advanced instruction in complex systems and AI methodologies
- **Entertainment**: Interactive storytelling, gaming applications, and immersive experiences

### 🛠️ Technology Stack

**PyScrAI** builds upon and extends industry-leading technologies including:

- **Google DeepMind's Concordia**: Core agent simulation framework
- **OpenRouter API**: OpenAI-compatible cloud language model endpoints
- **LM Studio**: Local generative language model deployment
- **LangGraph**: Advanced agent workflow orchestration
- **Sentence-Transformers**: Semantic embedding and retrieval systems
- **PyTorch**: Deep learning infrastructure
- **NVIDIA CUDA SDK**: Accelerated local model inference

This robust technology foundation ensures scalable, performant, and reliable AI agent simulations across diverse application scenarios.

## 📁 Project Structure

```
pyscrai_workstation/
├── 📦 concordia/                    # Core Concordia framework
│   ├── agents/                      # Agent implementations and behaviors
│   ├── associative_memory/          # Memory systems for agents
│   ├── clocks/                      # Timing and synchronization
│   ├── components/                  # Reusable simulation components
│   ├── document/                    # Document management
│   ├── environment/                 # Simulation environments and engines
│   ├── examples/                    # Example simulations and tutorials
│   ├── language_model/              # LLM integrations (GPT, OpenRouter, LMStudio)
│   ├── prefabs/                     # Pre-built simulation templates
│   ├── thought_chains/              # Agent reasoning systems
│   └── utils/                       # Utility functions
│
├── 📦 pyscrai/                      # Core PyScrai package
│   ├── agentica/                    # Agent framework and tools
│   └── geo_mod/                     # Geopolitical simulation module
│       ├── prefabs/                 # Nation entities, moderators
│       ├── scenarios/               # Specific simulation configurations
│       ├── simulations/             # Runnable simulation scripts
│       └── utils/                   # Logging and helper functions
│
├── 📁 util/                         # Project utilities
│   ├── docs/                        # Documentation files
│   ├── scripts/                     # Automation and setup scripts
│   ├── setup/                       # Environment setup tools
│   └── storage/                     # Data storage utilities
│
├── 📄 .env                          # Environment configuration
├── 📄 pyproject.toml               # Python project configuration
├── 📄 setup.py                     # Package installation script
└── 📄 README.md                    # This file
```

## 🧩 Core Components

### Concordia Framework
Environmental and scenario driven advanced agent cognitive simulation framework providing:

- **Agents**: Sophisticated AI agents with memory, personality, and decision-making
- **Associative Memory**: Long-term knowledge storage and retrieval systems
- **Game Clocks**: Temporal control and synchronization for simulations
- **Components**: Modular building blocks for agent behaviors
- **Environment Engine**: Simulation runtime and interaction management
- **Language Models**: Integration with various LLM providers
- **Thought Chains**: Complex reasoning and decision-making processes

### PyScrai Package

#### Agentica
Tools and frameworks for building AI agents with:
- **Agent lifecycle management**
- **Behavior configuration systems**
- **Communication protocols**
- **Performance monitoring**

#### Geo_mod
Specialized geopolitical simulation framework featuring:
- **Nation Entities**: Configurable country representatives with goals and contexts
- **Debate Moderators**: AI-powered mediation for international discussions
- **Scenario Configurations**: Pre-built templates for specific geopolitical situations
- **Turn-based Simulations**: Structured interaction protocols

## 📚 Documentation System - WIP


## 🔧 Environment Configuration

The project uses environment variables for configuration:

- **Language Models**: OpenRouter API, LMStudio integration
- **Storage Paths**: Document and data storage locations
- **API Tokens**: Hugging Face, GitHub integration
- **Model Selection**: Configurable AI model backends

. **Environment Setup**: Configure your `.env` file with API keys and paths
. **Virtual Environment**: Activate the `.pyscrai` virtual environment





### **Example Usage**: 
#### Agentica 
Running a multiturn interactive llm chatbot with Pyscrai.Agentica: 
```bash
 python -m pyscrai.agentica.cli --template chat --verbose  --interactive  scenarios in `pyscrai/geo_mod scenarios/
```
 - To run the rag agent, swap out 
 `--template chat`
 with 
 `--template rag` 

 #### Geo_Mod  - WIP


## 📋 Project Status

This is an active development workstation with ongoing enhancements to:
- Geopolitical simulation capabilities
- Improving scope, domain and type of simulation capabilities.
- Advanced, human-like cognitive agent systems. 
- Integration with more third-party tools, services and API's. 
- The documentation system is currently under `HEAVY development`. 
---