# Refactoring Strategies

The current code base mixes top level scripts with reusable modules. The following actions are recommended:

1. **Public APIs** – expose classes and helpers through package `__init__` files (implemented in this commit).
2. **Encapsulate Embedding Logic** – convert `embedder` into a class and allow token injection (implemented via `HFEmbedder`).
3. **Decouple Printing from Engines** – `DebateEngine.run_loop` prints directly; instead return events and let the caller handle I/O (implemented via returned event logs).
4. **Scenario Configuration** – move hard coded scenario parameters to data files or configuration objects.
5. **Testing** – introduce unit tests for components and engine logic to ensure reliability during future expansions.
