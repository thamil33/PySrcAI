### What Was Fixed

- The engine now creates an `ActionSpec` object for each agent's turn and passes it to `agent.act(action_spec)`.
- The archon also receives an `ActionSpec` when it acts.

### What This Means

- **`ActionSpec`** is a data structure that tells the agent what kind of action to take. It includes:
  - `call_to_action`: A prompt or instruction for the agent (e.g., "What do you do?").
  - `output_type`: The expected type of response (e.g., free text, choice, number).
  - `tag`: A label for the action (e.g., "demo", "archon_analysis").

- **Why is this needed?**
  - Your agent classes are designed to require this context for every action. It allows you to specify different prompts, action types, or tags for different situations.
  - Without it, the agent doesn't know what kind of action to take, so calling `act()` without an `ActionSpec` causes an error.

- **In practice:**  
  - The engine now provides a generic prompt for each agent's turn, so the simulation can run without errors.
  - You can later customize the `ActionSpec` for more complex scenarios (e.g., different prompts for different agents or steps).

--


**Customizing the `ActionSpec`** for each agent or scenario is best handled within a module (such as `mod_debate`, `mod_scenario`, etc.) or via the agent/archon configuration. Here’s how it typically works:

---

## How ActionSpec Customization Fits In

- **Core Engine:**  
  Provides the basic simulation loop and a default, generic `ActionSpec` (as you have now).

- **Modules (e.g., mod_debate, mod_scenario):**  
  - Can define their own logic for generating `ActionSpec` objects based on the scenario, agent role, or step in the simulation.
  - Can use config files to specify prompts, action types, or tags for each agent or situation.
  - The engine can be extended or subclassed in a module to inject this custom logic.

- **Agent/Archon Config:**  
  - You can add fields to your config (YAML/JSON) that specify what kind of actions, prompts, or tags should be used for each agent or role.
  - The factory or module-specific engine can read these and generate the appropriate `ActionSpec` at runtime.

---

## Example (for later)

```yaml
agents:
  - name: Alice
    type: actor
    action_prompts:
      - "You are debating. What is your opening statement?"
      - "How do you respond to your opponent's argument?"
    # ...
```
Then, in your module or engine, you’d use these prompts to build the `ActionSpec` for each turn.

---

## Summary

- **For now:** The generic `ActionSpec` in the engine is perfect for demos and development.
- **Later:** You can add module-specific or config-driven customization for richer, scenario-specific agent behavior.

You’re thinking about this exactly the right way!  
Let me know when you want to tackle this, and I’ll help you design a flexible system for it.