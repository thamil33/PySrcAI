# PyScrAI v0.5.0 Release Blueprint
> **A Personal Note from @thamil33**  
> _[Skip this story and jump to the technical details.](#major-goals-for-v050)_


[thoughtful] As a lifelong independent tech enthusiast, I’ve always dreamed big—often far beyond my skillset, hardware, or budget. [chuckles] Over the past few years, being immersed in the world of LLMs has been nothing short of transformative. [reflective] I’ve watched the impossible become reality: from the days when people said local models would never run on consumer hardware, to now—where even our phones can run impressive models, and open-source innovation is thriving.

[inspired] The rise of LLMs and agent frameworks has truly leveled the playing field, empowering individuals like me. [warm] I’ve learned more in these recent years than in my entire life, and for the first time, many of my development dreams are within reach.

[somber] Of course, reality has its challenges. [wry] GPU prices skyrocketed, and providers began raising costs and restricting free access. [pause] Just as things seemed out of reach, I discovered OpenRouter—a platform offering a wide selection of models, an OpenAI-compatible API, and a Python SDK. [relieved] Most importantly, they provide access to genuinely free models (with minimal credit requirements and reasonable limitations). [hopeful] This was the equalizer I needed.

[curious] But after months of experimentation, I realized that most frameworks only supported simple chatbots with OpenRouter’s free models. [determined] So, I set out to build something new: a framework inspired by the best features of major agent systems like Agno, AutoGen, and CrewAI, but designed from the ground up for full compatibility with OpenRouter and free models.

[excited] This is what inspired PyScrAI. [honest] While it doesn’t yet support free models for every use case, the goal is to maximize “free” wherever possible. [focused] That’s why I’m currently focused on OpenRouter as the primary provider. [growing confidence] What began as a hopeful experiment has grown into a vision to bring new capabilities to the open-source community—features not yet seen elsewhere. [realistic] There’s still a long road ahead. [smile] This pre-release is essentially a robust RAG agent for the CLI, with tests and examples, but it’s a solid proof of concept and a strong foundation.

[grateful] PyScrAI will always remain open-source, transparent, and free—my way of giving back to a community that has empowered and inspired me for so long. [gentle] I hope it can do the same for others.

---

## Overview
This document summarizes the progress, milestones, and key decisions made in the development of PyScrAI up to the 0.5.0 release. It serves as a blueprint and changelog for the project, capturing the goals achieved and the technical solutions implemented.

## Primary Motivation for PyScrAI 


---

## Major Goals for v0.5.0
- [x] **Robust RAG (Retrieval-Augmented Generation) Pipeline**
- [x] **OpenRouter LLM Integration**
- [x] **Local and API-based Embedding Support**
- [x] **LangChain Documentation Ingestion and Querying**
- [x] **Flexible Configuration System (YAML templates)**
- [x] **Provider Routing and Free Model Support**
- [x] **Comprehensive Testing and Debugging Utilities**
- [x] **Project Hygiene: Cache Cleaning, Test Organization**

---

## Key Progress and Solutions

### 1. RAG Pipeline
- Implemented a default RAG setup script (`recipes/default_rag_setup.py`) for ingesting and querying documentation.
- Supports both local sentence-transformers.
- Vector storage is flexible, supporting temporary and persistent directories.

### 2. OpenRouter LLM Integration
- Integrated OpenRouter as a primary LLM provider, with support for both free and paid models.
- Identified and resolved 404 errors when using free models with complex provider routing.
- Ensured compatibility by using simplified provider routing for free models (e.g., `{ "sort": "price" }`).
- Added direct and streaming LLM invocation examples.

### 3. Embedding Support
- Local embedding via `all-MiniLM-L6-v2` (sentence-transformers) for fast, API-free operation.
- Configurable via YAML templates and runtime overrides.

### 4. Documentation Ingestion & Querying
- Automated ingestion of LangChain documentation (or sample docs if not present).
- Demonstrated RAG queries with context retrieval and answer generation.
- Provided clear user feedback for missing docs and ingestion errors.

### 5. Configuration System
- Centralized YAML-based configuration templates (see `pyscrai/config/templates/`).
- Easy override of model, embedding, and vector DB settings in scripts.
- Environment variable support for API keys.

### 6. Provider Routing & Free Model Support
- Discovered OpenRouter API limitations with free models and complex routing.
- Documented and enforced correct provider routing for free models.
- Added targeted tests to verify model/routing combinations and prevent regressions.

### 7. Testing & Debugging
- Created direct LLM invocation tests (`test_openrouter_direct.py`).
- Added provider routing tests (`test_provider_routing.py`) and moved them to the `tests/` folder for better organization.
- Provided scripts for cache cleanup (`util/scripts/clean_project_cache.py`).

### 8. Project Hygiene
- Automated removal of all Python cache files and folders.
- Consolidated test scripts into the `tests/` directory.
- Maintained a clean and organized project root.

---

## Next Steps (Post-0.5.0)
- Currently Dreaming about this! -

---

## Acknowledgements
- Thanks to all contributors and testers for helping reach this milestone! -Tyler 

---

**PyScrAI v0.5.0 is already a robust, flexible, and well-tested foundation for RAG and LLM-powered applications. This is only the beggining...**




- Run fastapi fastapi with html Webapp UI - pyscrai\interface\api\main.py

``` python

uvicorn pyscrai.interface.api.main:app --host 127.0.0.1 --port 8000 --reload

```