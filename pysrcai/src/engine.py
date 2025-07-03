from pysrcai.src.agents import ActionSpec, OutputType

class SimulationEngine:
    """
    Base class for simulation engines. Handles the simulation lifecycle and agent management.
    """
    def __init__(self, agents=None, state=None):
        self.agents = agents if agents is not None else []
        self.state = state if state is not None else {}
        self.scenario_state = self.state.copy()  # scenario_state is the evolving state
        self.running = False

    def initialize(self):
        """Prepare the simulation (reset state, initialize agents, etc.)."""
        pass

    def step(self):
        """Advance the simulation by one step/tick."""
        pass

    def run(self, steps=None):
        """Run the simulation for a given number of steps, or until stopped."""
        self.running = True
        self.initialize()
        step_count = 0
        while self.running and (steps is None or step_count < steps):
            self.step()
            step_count += 1

    def shutdown(self):
        """Clean up resources and stop the simulation."""
        self.running = False

class SequentialEngine(SimulationEngine):
    """
    Concrete engine for sequential, turn-based agent interactions with optional archon observation.
    """
    def __init__(self, agents=None, archon=None, state=None):
        super().__init__(agents=agents, state=state)
        self.archon = archon
        self.turn = 0

    def initialize(self):
        self.turn = 0
        # Optionally, reset agent and archon state here
        print("[Engine] Initialized. Agents:", [a.name for a in self.agents])
        if self.archon:
            print(f"[Engine] Archon: {self.archon.name}")

    def step(self):
        print(f"\n[Engine] Step {self.turn+1}")
        # Example: add a conversation log to scenario_state
        if 'conversation_log' not in self.scenario_state:
            self.scenario_state['conversation_log'] = []
        for agent in self.agents:
            if hasattr(agent, 'observe'):
                agent.observe(f"Turn {self.turn+1}: It's your turn to act.")
            if hasattr(agent, 'act'):
                action_spec = ActionSpec(
                    call_to_action=f"Step {self.turn+1}: What do you do?",
                    output_type=OutputType.FREE,
                    tag="demo"
                )
                action = agent.act(action_spec)
                print(f"{agent.name} acts: {action}")
                # Log the action in scenario_state
                self.scenario_state['conversation_log'].append({
                    'turn': self.turn+1,
                    'agent': agent.name,
                    'action': action
                })
                if self.archon and hasattr(self.archon, 'observe'):
                    self.archon.observe(f"{agent.name} acted: {action}")
        if self.archon and hasattr(self.archon, 'act'):
            analysis = self.archon.act(ActionSpec(
                call_to_action=f"Step {self.turn+1}: Analyze the round.",
                output_type=OutputType.FREE,
                tag="archon_analysis"
            ))
            print(f"[Archon] {self.archon.name} analyzes: {analysis}")
            self.scenario_state['conversation_log'].append({
                'turn': self.turn+1,
                'agent': self.archon.name,
                'action': analysis
            })
        self.turn += 1 