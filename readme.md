# PySrcAI

## **Update** `v0.7.1` Released

- Released 06/25/2025: Major step forward in the development of PySrcAI

## 🎯 Project Overview

**`PySrcAI` Pronounced _Pie-Scry_, is a comprehensive framework engineered for broad multi-domain applications while maintaining exceptional quality, performance, and modularity. The platform enables seamless configuration for highly controlled, specialized use cases across diverse sectors and industries.**

**`PySrcAI` delivers advanced tooling for creating and deploying sophisticated agent-based scenarios, with particular emphasis on simulations and cognitive reasoning AI applications. The platform integrates multiple leading frameworks and technologies to provide a unified development environment for building enterprise-grade multi-agent systems, simulations, and AI solutions.**

### 🔬 Target Application Domains

- **Academic Research**: Political science, international relations, and AI safety research
- **Policy Analysis**: Strategic scenario planning for government agencies and NGOs  
- **AI Development**: Multi-agent system research, development, and deployment
- **Educational**: Advanced instruction in complex systems and AI methodologies
- **Entertainment**: Interactive storytelling, gaming applications, and immersive experiences

### 🛠️ Technology Stack

**PySrcAI** builds upon and extends industry-leading technologies including:

- **Google DeepMind's Concordia**: Core agent simulation framework
- **OpenRouter API**: OpenAI-compatible cloud language model endpoints
- **LM Studio**: Local generative language model deployment
- **LangGraph**: Advanced agent workflow orchestration
- **Sentence-Transformers**: Semantic embedding and retrieval systems
- **PyTorch**: Deep learning infrastructure
- **NVIDIA CUDA SDK**: Accelerated local model inference

This robust technology foundation ensures scalable, performant, and reliable AI agent simulations across diverse application scenarios.

## 📁 Project Structure

pysrcai_workstation/
pysrcai_workstation/
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
├── 📦 pysrcai/                      # Core PySrcAI package
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

### PySrcAI Package

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

## 🔧 Environment Setup & Config

### Python Environment Setup (with UV & CUDA/PyTorch Support)

These steps will set up a fresh Python 3.12 environment using [UV](https://github.com/astral-sh/uv) for dependency management, and install all requirements (including CUDA/PyTorch support) for local model inference.

#### 1. Create and Activate Virtual Environment

(Change to your directory of python 3.12)

```powershell
C:\Users\tyler\AppData\Local\Programs\Python\Python312\python.exe -m venv .pysrcai 
.\.pysrcai\Scripts\Activate
```

#### 2. Upgrade pip (recommended)

```powershell
python -m pip install --upgrade pip
```

#### 3. Install UV (fast Python package manager)

```powershell
pip install uv
```

#### 4. Install All Project Requirements (including CUDA/PyTorch)

```powershell
uv pip install -r requirements.txt
```

#### 5. (Optional) Install Project in Editable/Dev Mode

```powershell
uv pip install -e .
```

You are now ready to use PySrcAI with full CUDA/PyTorch support!

## **Example Usage**

### Agentica Interactive CLI

Running a multi turn interactive llm chatbot with PySrcAI.Agentica:

```bash
 python -m pysrcai.agentica.cli --verbose  --interactive
```

- To run the rag agent or any other template, pass the --template arg

```bash
python -m pysrcai.agentica.cli --template rag
```
  
### Geo_Mod  - WIP

## 📋 Project Status

This is an active development workstation with ongoing enhancements to:

- Geopolitical simulation capabilities
- Improving scope, domain and type of simulation capabilities.
- Advanced, human-like cognitive agent systems.
- Integration with more third-party tools, services and APIs.
- The documentation system is currently under `HEAVY development`.
