class SimulationEngine:
    """
    Base class for simulation engines. Handles the simulation lifecycle and agent management.
    """
    def __init__(self, agents=None, state=None):
        self.agents = agents if agents is not None else []
        self.state = state if state is not None else {}
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