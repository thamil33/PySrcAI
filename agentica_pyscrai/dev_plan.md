# DevPlan

Mark out tasks when accomplished, if in doubt leave unchecked. 

[x] 1. Directory & Package Setup
Create the new top-level package agentica_pyscrai.
Inside, add subpackages: cli, config, agents, adapters, tests, docs.
Provide minimal __init__.py files in each.

[x] 2. Configuration System
Design dataclasses (or pydantic models) for:

[x] ModelConfig: LLM settings for OpenRouter and LMStudio.

[x] EmbeddingConfig: HuggingFace and sentence-transformers options.

[x] VectorDBConfig: Chroma settings (collection name, persist path).

[x] AgentConfig: high-level RAG agent parameters.

[x] Implement YAML loading with environment variable interpolation for API keys.

[x] Place default config examples under agentica_pyscrai/config/templates.

[x] 3. Embedding Adapters
Implement a wrapper class for HuggingFace API embeddings.

Implement a wrapper for local sentence-transformers models.

Ensure both conform to LangChain's Embeddings interface.

[x] 4. LLM Adapters
[x] Create an adapter for OpenRouter API LLMs (LangChain LLM interface).
[x] Add a skeleton LMStudio adapter (placeholder for future development).
[x] Allow configuration to swap between these backends.

5. Vector Store Integration
Use LangChain's Chroma wrapper for persistent vector storage.

Support ingesting documents and similarity search.

Expose vector store path, collection name, and other parameters via config.

6. Document Ingestion
Implement loaders for directories containing text, Markdown, or JSON.

Provide chunking using LangChain text splitters (configurable size/overlap).

Optionally add simple preprocessing hooks.

7. Base Agent & Builder
Define an abstract BaseAgent with methods:

ingest(doc_paths)

query(question)

clear_store()

interactive_loop() (for CLI use)

Implement a concrete RAGAgent leveraging LangChain's retrieval QA chain.

Provide a builder/factory to assemble the agent from configuration.

8. Interactive CLI
[x] Add a console entry point (e.g., python -m agentica_pyscrai.cli).

[x] Commands/options:
[x] --config <yaml> to load configuration.
[ ] --ingest <path> to ingest docs.
[ ] --query "<question>" to run once.
[ ] --interactive to start a REPL loop.
[ ] --clear to wipe the vector store.

Keep CLI style consistent with existing pyscrai tool.

9. Default RAG Setup
[x] Provide an example config for ingesting LangChain docs and module docs.

Include a script or instructions to download LangChain documentation.

Verify the default config works with minimal steps.

10. Pytests
[x] Unit tests for configuration loader and data classes.

[x] Embedding adapter initialization.

[ ] Base agent's ingest/query logic (mocking external services).

[x] Ensure pytest can run from the repo root.

11. Documentation
Populate docs/ with:

Setup and usage instructions.

[x] Config file examples.

Brief explanation of how to extend for LangMem or Concordia integration later.

12. Future Enhancements
Create TODOs or issue templates for:

Integrating LangMem once it matures.

UI dashboard for managing agents.

Concordia framework merging or interfacing.