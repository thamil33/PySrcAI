# Debate Engine Development Plan (Adjusted per Scenario)

## 1. Objectives

- Build a **modular, scalable debate simulator** in `pysrcai/pysrcai/debate`.
- Reference Concordia and `geo_mod` patterns, but **do not depend on or import from their code**; implement all logic in this module’s own `src` directory.
- Initial scenario: Two participants, with an orchestrator (moderator, non-participant) overseeing a philosophical/political/religious debate.

---

## 2. Architecture & Component Plan

### **A. Core Components**

1. **DebateSession**

   - Orchestrates the debate lifecycle.
   - Holds references to participants, orchestrator, debate rules, and logger.

2. **Participant**

   - Base class/interface for entities that take part in the debate.
   - Subclasses for various participant types (AI/human/scripted, etc).

3. **Orchestrator**

   - Acts as the moderator; not a participant.
   - Manages turn order, rules, and enforces debate flow.

4. **DebateRules**

   - Encapsulates round structure, allowed moves, turn limits, etc.
   - Flexible for custom rulesets (future extensibility).

5. **Logger/Recorder**

   - Records all actions/events for transparency, debugging, and output.

6. **OutputRenderer** (optional for first iteration)

   - Translates logs to readable format (text/HTML/etc).

---

### **B. Support/Utility Modules**

- **Message/Argument Model:** Structured representation of debate arguments.
- **Config:** Handles parameterization (in-code or config file, e.g., topic, rounds).

---

### **C. File Structure**

Lets try to utilize the existing structure in place, although there is certainly room for adjustment as we reach further stages of development this can be decided upon more rigidly in the future. 

---

## 3. Milestones

**Phase 1: Scaffolding**

- Class stubs & docstrings for all major components.
- Unit test skeleton.

**Phase 2: Minimal Working Example**

- Implement simple two-participant debate flow (one topic, basic turn-taking, logging).
- Orchestrator enforces turns, time/round limits, and logs each statement.

**Phase 3: Modularization**

- Support alternative rules, participant logic, and orchestrator behaviors.
- Add more robust config and logging.

**Phase 4: Documentation/Examples**

- README and example scripts for usage/testing.

---

## 4. Design Principles

- **All code implemented locally in `pysrcai\pysrcai\debate\src` —no direct dependency on Concordia or geo\_mod source.
<!-- -However code, functions, classes, and even submodules can be essentially 'copied' from either the concordia or geo_mod sources as long as they are compatible with our modular system. (Especially early on in development).  -->
- **Separation of Concerns**: Debate flow, participant logic, and logging are modular and independent.
- **Extensibility**: Easy to add new debate formats, participant strategies, or logging styles.
- **Testability**: Core logic covered by unit tests.

---

## 5. Usage Example (Minimal Scenario)

```python
from pysrcai.debate.src.session import DebateSession
from pysrcai.debate.src.participant import BasicParticipant
from pysrcai.debate.src.orchestrator import Orchestrator
from pysrcai.debate.src.rules import SimpleRules

# Setup
participants = [BasicParticipant(name='Alice'), BasicParticipant(name='Bob')]
orchestrator = Orchestrator()
rules = SimpleRules(rounds=3, topic="Is free will an illusion?")

session = DebateSession(participants=participants, orchestrator=orchestrator, rules=rules)
session.run()
```

---

## 6. Next Steps

 Begin implementation with class scaffolding and minimal example.

---

**This plan reflects all requirements and adjustments discussed. Ready for your review!**

