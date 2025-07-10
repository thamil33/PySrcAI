"""Sequential engine implementation for PySrcAI.

This module provides a concrete engine for sequential, turn-based agent
interactions with optional archon observation.
"""

from pysrcai.core.engines.base_engine import SimulationEngine
from pysrcai.agents.environment.environment_components import EnvironmentContextComponent
from pysrcai.agents.environment.action_components import ActionProcessingComponent as ActionProcessor
from pysrcai.core.objects import create_environment_from_config, Environment

class SequentialEngine(SimulationEngine):
    def __init__(self, agents=None, archon=None, state=None, config=None):
        super().__init__(agents=agents, state=state)
        self.archon = archon
        self.turn = 0
        self.config = config or {}

        # Initialize configuration
        self._init_config()

        # Initialize environment
        self._init_environment()

        # Initialize components
        self.context_generator = EnvironmentContextComponent(self.environment)
        self.action_processor = ActionProcessor(self.environment)

    def _init_config(self):
        """Initialize configuration settings."""
        self.response_word_limit = None
        engine_cfg = self.config.get('engine', {}) if self.config else {}
        if 'response_word_limit' in engine_cfg:
            self.response_word_limit = engine_cfg['response_word_limit']

    def _init_environment(self):
        """Initialize environment from config."""
        if "scenario" in self.config and "initial_state" in self.config["scenario"]:
            print("[Engine] Loading environment from config...")
            self.environment = create_environment_from_config(self.config)
            print(f"[Engine] Environment loaded. Active location: {self.environment.active_location}")
            print(f"[Engine] Available locations: {list(self.environment.locations.keys())}")
            print(f"[Engine] Available items: {list(self.environment.items.keys())}")
        else:
            print("[Engine] No scenario config found, creating empty environment")
            self.environment = Environment()

        # Add environment to scenario state
        self.scenario_state["environment"] = self.environment.to_dict()

    def initialize(self):
        """Initialize the simulation."""
        self.turn = 0
        print("[Engine] Initialized. Agents:", [a.name for a in self.agents])
        if self.archon:
            print(f"[Engine] Archon: {self.archon.name}")

        self._notify_agents_of_environment()

    def _notify_agents_of_environment(self):
        """Notify all agents and archon of the initial environment state."""
        if not self.environment.active_location:
            return

        active_location = self.environment.locations.get(self.environment.active_location)
        if not active_location:
            return

        desc = f"You are in the {active_location.name}. {active_location.description}"

        objects = [obj.name for obj in active_location.objects.values()]
        if objects:
            desc += f" You see the following objects: {', '.join(objects)}."
        else:
            desc += " There are no notable objects here."

        # Inform agents and archon
        for agent in self.agents:
            if hasattr(agent, 'observe'):
                agent.observe(desc)

        if self.archon and hasattr(self.archon, 'observe'):
            self.archon.observe(desc)

    def step(self):
        """Execute one simulation step."""
        print(f"\n[Engine] Step {self.turn+1}")

        # Initialize conversation log if needed
        if 'conversation_log' not in self.scenario_state:
            self.scenario_state['conversation_log'] = []

        # Process each agent's turn
        for agent in self.agents:
            self._process_agent_turn(agent)

        # Final archon analysis
        self._process_archon_analysis()

        self.turn += 1

    def _process_agent_turn(self, agent):
        """Process a single agent's turn."""
        # Create observation for agent
        inventory = self._get_agent_inventory(agent)
        observation = self.context_generator.get_agent_observation(
            agent.name, self.turn + 1, inventory
        )

        if hasattr(agent, 'observe'):
            agent.observe(observation)

        # Process action
        action = agent.act()
        self.action_processor.process_action(agent, action)

    def _process_archon_analysis(self):
        """Process the archon's analysis at the end of the step."""
        if self.archon:
            analysis = self.archon.analyze(self.scenario_state)
            self.scenario_state['conversation_log'].append(analysis)
            
    def _get_agent_inventory(self, agent):
        """Get items in an agent's inventory.
        
        Args:
            agent: The agent whose inventory to retrieve
            
        Returns:
            List of item IDs in the agent's inventory
        """
        inventory = []
        for item_id, item in self.environment.items.items():
            location = item.properties.get("location", "")
            if location == f"inventory_{agent.name}":
                inventory.append(item_id)
        return inventory
