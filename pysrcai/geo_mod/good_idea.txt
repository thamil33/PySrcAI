# Geopolitical Simulation Framework in Concordia

## Core Concept

This project aims to create a flexible and extensible simulation framework within Concordia to model geopolitical interactions, conflicts, and public sentiment. By representing nation-states and their citizens as distinct entities, we can explore the complex interplay between high-level state strategy and the micro-level psychological impact on individuals.

The framework will be built in phases, starting with a simple, direct conflict and progressively adding layers of complexity.

---

## Phase 1: The Foundational Debate

The initial scenario will be a simple, two-party debate to establish the core mechanics.

*   **Scenario:** A UN-style moderated debate between Russia and Ukraine.
*   **Actors:**
    *   **Entity 1: Russia.** Programmed with its publicly stated geopolitical goals, historical context, and desired outcomes regarding the conflict.
    *   **Entity 2: Ukraine.** Programmed with its own goals for sovereignty, security, and desired outcomes.
    *   **Game Master 1: The Moderator.** A neutral GM that facilitates a turn-based debate, ensuring each side gets a set number of turns (e.g., two each) to present their case.
*   **Objective:** To create a baseline interaction where each country-entity clearly articulates its position. This tests the ability of the LLM to role-play a nation-state accurately.
*   **Modularity:** The "Country" entity will be designed as a flexible prefab, allowing us to easily swap in different nations by changing its parameters (e.g., name, goals, history).

---

## Phase 2: Adjudication and Strategic Dialogue

This phase introduces a layer of evaluation and more complex interaction.

*   **New Actor:**
    *   **Game Master 2: The Judge.** An entity representing an international body or a panel of neutral observers.
*   **New Mechanics:**
    *   After the debate, the Judge GM will analyze the arguments from each side.
    *   It will "score" the debate based on predefined criteria, such as:
        *   Persuasiveness of the argument.
        *   Alignment with stated goals.
        *   Consistency and coherence.
    *   The Judge will declare a "winner" of the debate, providing a qualitative summary of its decision. This introduces a clear success/failure metric.

---

## Phase 3: Multi-Actor Strategic Simulation

Expand the simulation to a multi-polar world with more complex dynamics.

*   **New Actors:**
    *   Introduce other influential nation-entities (e.g., USA, China, Germany) or blocs (e.g., NATO).
*   **New Mechanics:**
    *   Move beyond simple debate to strategic actions. Entities can form alliances, propose treaties, impose sanctions, or offer aid.
    *   Introduce a **World Events GM** that can inject unexpected events (e.g., an economic crisis, a natural disaster, a technological breakthrough) that all nations must react to.

---

## Phase 4: The Macro-Micro Link (State and Citizen)

This is the most innovative phase, connecting state-level actions to the sentiment of the populace.

*   **New Actors:**
    *   For each nation-state, introduce a small cohort of **"Citizen" entities.**
    *   These citizens are not strategic actors but are **reactors**. Their memories and psychological states are shaped by the high-level simulation.
*   **New Mechanics:**
    *   The outcomes of the geopolitical simulation (e.g., a summary of the debate, news of a new treaty or sanction) are fed as observations to the Citizen entities.
    *   After these events, a **Survey GM** (similar to the `interviewer__GameMaster`) administers psychological questionnaires (e.g., DASS, or custom surveys on nationalism, hope, fear) to the citizens.
*   **Objective:** To measure the "psychological weather" of a nation. We can analyze how state-level decisions and international events impact the mental well-being and opinions of the average citizen, providing a powerful, two-level view of the simulated world.

This phased approach allows us to build a robust and insightful simulation framework, starting with a manageable scope and adding complexity incrementally. It sounds like an excellent project!
