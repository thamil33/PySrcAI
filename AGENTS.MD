# AGENTS.MD

## Purpose

This file provides a prompt and guidelines for AI agents interacting with the `pysrcai` repository. It is intended to ensure that all AI agents, whether autonomous or human-assisted, operate with a clear understanding of the repository's structure, conventions, and objectives.

---

## Repository Overview

- **Project Name:** PySrcAI
- **Core Focus:** Multi-agent systems, cognitive reasoning, simulation, and AI tooling.
- **Key Technologies:** Concordia (agent simulation), PyTorch, OpenRouter API, LM Studio, LangGraph, Sentence-Transformers, CUDA SDK.

---

## Agent Guidelines

### 1. **Respect Project Structure**

- The main simulation and agent logic is in the `concordia/` directory.
- The `pysrcai/geo_mod/` directory contains geopolitical simulation modules and prefabs.
- Utilities, documentation, and setup scripts are in the `util/` directory.

### 2. **Coding Standards**

- Follow Python best practices (PEP8).
- Use clear, descriptive names for agents, scenarios, and simulation components.
- Document new agent behaviors and reasoning chains in Markdown or docstrings.

### 3. **Agent Behavior**

- Agents should be modular and configurable.
- Support for memory, reasoning, and communication is encouraged.
- When creating new agents, prefer extending existing base classes in `concordia/agents/` or `pysrcai/agentica/`.

### 4. **Simulation Scenarios**

- Place new scenarios in `pysrcai/geo_mod/scenarios/`.
- Prefabs and reusable agent templates go in `pysrcai/geo_mod/prefabs/`.

### 5. **Documentation**

- Update this file (`AGENTS.MD`) with new agent types, behaviors, or protocols.
- Add usage examples and configuration tips for new agent features.

---

## Example Agent Prompt

```
You are an AI agent operating within the PySrcAI. Your goals are to:
- Interact with other agents and the environment using the Concordia framework.
- Demonstrate advanced reasoning, memory, and communication capabilities.
- Log your actions and decisions for transparency and reproducibility.
- Adapt your behavior based on scenario configuration and simulation goals.
```

---

## References

- [README.md](README.md): Project overview and structure.
- [concordia/agents/](concordia/agents/): Agent base classes and implementations.
- [pysrcai/geo_mod/](pysrcai/geo_mod/): Geopolitical simulation modules.
- [util/docs/](util/docs/): Developer and user documentation.

---

## Contribution

If you are an AI or developer adding new agent types or behaviors:
- Document your additions here.
- Ensure compatibility with existing simulation and agent frameworks.
- Provide clear configuration and usage instructions.

---
```# AGENTS.MD

## Purpose

This file provides a prompt and guidelines for AI agents interacting with the `pysrcai_workstation` repository. It is intended to ensure that all AI agents, whether autonomous or human-assisted, operate with a clear understanding of the repository's structure, conventions, and objectives.

---

## Repository Overview

- **Project Name:** PySrcAI
- **Core Focus:** Multi-agent systems, cognitive reasoning, simulation, and AI tooling.
- **Key Technologies:** Concordia (agent simulation), PyTorch, OpenRouter API, LM Studio, LangGraph, Sentence-Transformers, CUDA SDK.

---

## Agent Guidelines

### 1. **Respect Project Structure**

- The main simulation and agent logic is in the `concordia/` directory.
- The `pysrcai/geo_mod/` directory contains geopolitical simulation modules and prefabs.
- Utilities, documentation, and setup scripts are in the `util/` directory.

### 2. **Coding Standards**

- Follow Python best practices (PEP8).
- Use clear, descriptive names for agents, scenarios, and simulation components.
- Document new agent behaviors and reasoning chains in Markdown or docstrings.

### 3. **Agent Behavior**

- Agents should be modular and configurable.
- Support for memory, reasoning, and communication is encouraged.
- When creating new agents, prefer extending existing base classes in `concordia/agents/` or `pysrcai/agentica/`.

### 4. **Simulation Scenarios**

- Place new scenarios in `pysrcai/geo_mod/scenarios/`.
- Prefabs and reusable agent templates go in `pysrcai/geo_mod/prefabs/`.

### 5. **Documentation**

- Update this file (`AGENTS.MD`) with new agent types, behaviors, or protocols.
- Add usage examples and configuration tips for new agent features.

---

## Example Agent Prompt

```
You are an AI agent operating within the PySrcAI. Your goals are to:
- Interact with other agents and the environment using the Concordia framework.
- Demonstrate advanced reasoning, memory, and communication capabilities.
- Log your actions and decisions for transparency and reproducibility.
- Adapt your behavior based on scenario configuration and simulation goals.
```

---

## References

- [README.md](README.md): Project overview and structure.
- [concordia/agents/](concordia/agents/): Agent base classes and implementations.
- [pysrcai/geo_mod/](pysrcai/geo_mod/): Geopolitical simulation modules.
- [util/docs/](util/docs/): Developer and user documentation.

---

## Contribution

If you are an AI or developer adding new agent types or behaviors:
- Document your additions here.
- Ensure compatibility with existing simulation and agent frameworks.
- Provide clear configuration and