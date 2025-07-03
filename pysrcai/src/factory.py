from .engine import SimulationEngine

class SimulationFactory:
    """
    Base class for simulation factories. Responsible for creating engines and agents from configuration.
    """
    def create_engine(self, config):
        """
        Create and return a SimulationEngine instance based on the provided config.
        """
        # Example: config could specify agent types, initial state, engine type, etc.
        agents = config.get('agents', [])
        state = config.get('state', {})
        return SimulationEngine(agents=agents, state=state) 