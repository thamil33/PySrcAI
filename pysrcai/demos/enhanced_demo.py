"""Enhanced demo using the modular PySrcAI framework.

This script demonstrates the enhanced capabilities of the PySrcAI framework
using the new module system for a debate scenario.
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path so we can import pysrcai modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pysrcai.config.config_loader import load_config
from pysrcai.core.enhanced_engine import EnhancedSimulationEngine
from pysrcai.agents.environment.enhanced_minimal import EnhancedMinimalEnvironment
from pysrcai.modules.debate.debate_module import DebateModule
from pysrcai.core.event_system import event_bus
from pysrcai.core.state_management import JsonSerializableStateContainer


# Default config path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'prefabs', 'enhanced_minimal.yaml')


class TeeOutput:
    """Captures output and writes to both console and file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate writing

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


class ModularDebateEngine(EnhancedSimulationEngine):
    """Enhanced engine that uses the module system for debates."""
    
    def __init__(self, config: dict):
        """Initialize the modular debate engine."""
        super().__init__(config)
        
        # Store debate-specific config
        self.word_limit = config.get("engine", {}).get("response_word_limit", 75)
        self.debate_module = None
        self.environment = None
        
        # Set up state container
        initial_state = {
            "turn": 0,
            "phase": "opening",
            "whiteboard": []
        }
        self.state_container = JsonSerializableStateContainer(initial_state)
    
    def setup(self):
        """Set up the engine with modules."""
        # Load environment module
        self.environment = EnhancedMinimalEnvironment()
        self.environment.initialize({
            "initial_notes": self.config.get("scenario", {}).get("initial_state", {}).get("whiteboard", [])
        })
        
        # Load debate module
        self.debate_module = DebateModule()
        self.debate_module.initialize({
            "debate_topic": self.config.get("scenario", {}).get("debate_topic", "Should we adopt a four-day work week?"),
            "word_limit": self.word_limit,
            "engine": self
        })
        
        # Register for events
        event_bus.subscribe("debate_complete", self._handle_debate_complete)
        event_bus.subscribe("request_moderation", self._handle_request_moderation)
        
        print("[Engine] Modules initialized")
    
    def run(self, steps=None):
        """Run the simulation for a specified number of steps."""
        if steps is None:
            steps = self.config.get("engine", {}).get("steps", 6)
            
        self.setup()
        
        print(f"[Engine] Running enhanced debate simulation for {steps} steps...")
        print(f"[Engine] Debate topic: {self.debate_module.get_debate_topic()}")
        print(f"[Engine] Word limit per response: {self.word_limit}")
        
        for _ in range(steps):
            if self.state == self.state.TERMINATED:
                break
                
            result = self.step()
            if result.get("terminated", False):
                print("[Engine] Simulation terminated.")
                break
                
        print("[Engine] Simulation complete.")
    
    def step(self):
        """Run a single simulation step."""
        self.turn += 1
        print(f"\n[Engine] Step {self.turn}")
        
        # Update state
        self.state_container.update(
            lambda state: {**state, "turn": self.turn}
        )
        
        # Get current phase from debate module
        current_phase = self.debate_module.get_current_phase().value
        
        # Determine whose turn it is
        turn_index = (self.turn - 1) % len(self.agents)
        current_agent = self.agents[turn_index]
        
        # Get agent action
        action = self._get_agent_action(current_agent, current_phase)
        
        # Process action in environment
        self._process_action(current_agent, action)
        
        # Emit event that agent acted
        event_bus.publish_event("agent_acted", {
            "agent": current_agent.name,
            "action": action,
            "phase": current_phase,
            "turn": self.turn
        })
        
        # Check termination
        if self.state == self.state.TERMINATED:
            return {"terminated": True}
        
        return {"terminated": False}
    
    def _get_agent_action(self, agent, phase):
        """Get an action from an agent."""
        from pysrcai.agents.base.agent import ActionSpec, OutputType
        
        # Build prompt for agent
        prompt = self._build_agent_prompt(agent, phase)
        
        # Create action spec
        action_spec = ActionSpec(
            call_to_action=prompt,
            output_type=OutputType.FREE,
            tag='debate_response'
        )
        
        # Get response from agent
        response = agent.act(action_spec)
        
        # Parse action using the action parser
        from pysrcai.agents.environment.enhanced_minimal import EnhancedMinimalActionParser
        try:
            verb, content, exceeded_limit = EnhancedMinimalActionParser.parse(response)
            
            result = {
                "type": verb,
                "content": content,
                "valid": True
            }
            
            if exceeded_limit:
                word_count = len(content.split())
                result["warning"] = f"Response exceeded {self.word_limit}-word limit (got {word_count} words)."
                print(f"[Engine] Warning: {result['warning']}")
            
            return result
        except ValueError as e:
            print(f"[Engine] Invalid response format: {e}")
            return {
                "type": "invalid",
                "content": "Invalid response format. Please follow the template exactly.",
                "valid": False,
                "error": str(e)
            }
    
    def _build_agent_prompt(self, agent, phase):
        """Build a prompt for an agent."""
        agent_role = "Proponent" if agent == self.agents[0] else "Opponent"
        debate_topic = self.debate_module.get_debate_topic()
        word_limit = self.debate_module.get_word_limit()
        
        prompt_parts = [
            f"You are the {agent_role} in a debate on the topic: '{debate_topic}'",
            f"Current phase: {phase}",
            "Whiteboard contents:",
        ]
        
        for i, entry in enumerate(self.environment.snapshot()):
            prompt_parts.append(f"[{i+1}] {entry}")
            
        if phase == "opening":
            prompt_parts.append("\nPresent your opening argument in favor of your position.")
            action_template = "ACTION: propose"
        elif phase == "rebuttal":
            prompt_parts.append("\nRespond to your opponent's arguments.")
            action_template = "ACTION: respond"
        elif phase == "closing":
            prompt_parts.append("\nPresent your closing summary.")
            action_template = "ACTION: respond"
            
        prompt_parts.append(f"\nRespond using this format:")
        prompt_parts.append(f"{action_template}")
        prompt_parts.append(f"CONTENT: <your argument in {word_limit} words or less>")
        
        return "\n".join(prompt_parts)
    
    def _process_action(self, agent, action):
        """Process an agent's action."""
        agent_name = agent.name
        action_type = action.get("type", "")
        content = action.get("content", "")
        
        print(f"{agent_name} acts: {action_type}")
        print(content)
        
        # Add to environment whiteboard
        self.environment.add_note(f"{agent_name}: {content}")
    
    def _handle_debate_complete(self, event):
        """Handle debate complete event."""
        print("[Engine] Debate complete")
        self.terminate()
    
    def _handle_request_moderation(self, event):
        """Handle request for moderation."""
        if not self.archon:
            return
            
        mod_type = event.data.get("type")
        phase = event.data.get("phase")
        
        if mod_type == "assessment":
            self._get_moderator_assessment(phase)
        elif mod_type == "final":
            self._get_moderator_final()
    
    def _get_moderator_assessment(self, phase):
        """Get assessment from moderator."""
        from pysrcai.agents.base.agent import ActionSpec, OutputType
        
        prompt = self._build_moderator_prompt(phase)
        
        action_spec = ActionSpec(
            call_to_action=prompt,
            output_type=OutputType.FREE,
            tag='moderator_assessment'
        )
        
        response = self.archon.act(action_spec)
        assessment = self._parse_moderator_response(response)
        
        print(f"[Moderator] {assessment.get('content', '')}")
        self.environment.add_note(f"Moderator: {assessment.get('content', '')}")
    
    def _build_moderator_prompt(self, phase):
        """Build prompt for moderator."""
        prompt_parts = [
            "You are the Moderator in a debate.",
            f"Topic: '{self.debate_module.get_debate_topic()}'",
            f"Phase just completed: {phase}",
            "Whiteboard contents:",
        ]
        
        for i, entry in enumerate(self.environment.snapshot()):
            prompt_parts.append(f"[{i+1}] {entry}")
            
        prompt_parts.append("\nProvide a brief assessment of this phase of the debate.")
        prompt_parts.append("ACTION: respond")
        prompt_parts.append(f"CONTENT: <your assessment in {self.word_limit} words or less>")
        
        return "\n".join(prompt_parts)
    
    def _get_moderator_final(self):
        """Get final judgment from moderator."""
        from pysrcai.agents.base.agent import ActionSpec, OutputType
        
        prompt = self._build_final_prompt()
        
        action_spec = ActionSpec(
            call_to_action=prompt,
            output_type=OutputType.FREE,
            tag='moderator_final'
        )
        
        response = self.archon.act(action_spec)
        judgment = self._parse_moderator_response(response)
        
        print(f"[Moderator] Final Judgment: {judgment.get('content', '')}")
        self.environment.add_note(f"Moderator: {judgment.get('content', '')}")
        self.environment.add_note("SIMULATION END")
    
    def _build_final_prompt(self):
        """Build final prompt for moderator."""
        prompt_parts = [
            "You are the Moderator in a debate that has just concluded.",
            f"Topic: '{self.debate_module.get_debate_topic()}'",
            "All phases are complete.",
            "Whiteboard contents:",
        ]
        
        for i, entry in enumerate(self.environment.snapshot()):
            prompt_parts.append(f"[{i+1}] {entry}")
            
        prompt_parts.append("\nProvide your final judgment on the debate.")
        prompt_parts.append("Declare a winner (Proponent or Opponent) and explain your reasoning.")
        prompt_parts.append("ACTION: respond")
        prompt_parts.append(f"CONTENT: <your final judgment in {self.word_limit} words or less>")
        
        return "\n".join(prompt_parts)
    
    def _parse_moderator_response(self, response):
        """Parse moderator response."""
        lines = response.strip().split("\n")
        action_type = ""
        content = ""
        
        for line in lines:
            if line.startswith("ACTION:"):
                action_type = line.replace("ACTION:", "").strip()
            elif line.startswith("CONTENT:"):
                content = line.replace("CONTENT:", "").strip()
                
        return {
            "type": action_type,
            "content": content
        }


class EnhancedSimulationFactory:
    """Factory for creating enhanced simulations."""
    
    @staticmethod
    def create_engine(config):
        """Create a simulation engine from config."""
        # Create engine instance
        engine = ModularDebateEngine(config)
        
        # Create and register agents
        from pysrcai.llm.llm_components import create_language_model
        from pysrcai.agents.base.actor import Actor
        from pysrcai.agents.base.archon import Archon
        from pysrcai.llm.llm_components import ActorLLMComponent, ArchonLLMComponent
        
        # Create agents
        for agent_config in config.get("agents", []):
            name = agent_config.get("name")
            agent_type = agent_config.get("type")
            
            if agent_type == "actor":
                # Create language model
                llm_type = agent_config.get("llm", "mock")
                llm = create_language_model(llm_type)
                
                # Create acting component
                acting_component = ActorLLMComponent(llm)
                
                # Create agent with the acting component
                agent = Actor(
                    agent_name=name,
                    act_component=acting_component,
                    personality_traits=agent_config.get("personality", {})
                )
                
                # Register with engine
                engine.register_agent(agent)
                
            elif agent_type == "archon":
                # Create language model
                llm_type = agent_config.get("llm", "mock")
                llm = create_language_model(llm_type)
                
                # Create acting component
                acting_component = ArchonLLMComponent(llm)
                
                # Create archon with the acting component
                authority_level = agent_config.get("authority_level", 0)
                archon = Archon(
                    agent_name=name,
                    act_component=acting_component,
                    authority_level=authority_level
                )
                
                # Register with engine
                engine.register_archon(archon)
        
        # Return the engine and steps
        steps = config.get("engine", {}).get("steps", 6)
        return engine, steps


def main():
    """Main function to run the enhanced demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the PySrcAI Enhanced Simulation')
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG_PATH,
                      help='Path to the configuration YAML file')
    args = parser.parse_args()
    
    config_path = args.config
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"pysrcai/data/logs/ENHANCED_LOG_{timestamp}.txt"
    
    # Ensure logs directory exists
    os.makedirs("pysrcai/data/logs", exist_ok=True)
    
    # Set up output capture
    tee = TeeOutput(log_filename)
    old_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print("=== PySrcAI Enhanced Demo ===")
        print(f"Output being saved to: {log_filename}")
        print(f"Using configuration: {config_path}")
        print()
        
        # Load scenario config
        config = load_config(config_path)
        
        # Enable async event processing
        event_bus.enable_async()
        
        # Create simulation using factory
        factory = EnhancedSimulationFactory()
        engine, steps = factory.create_engine(config)
        
        print(f"\n[Engine] Running for {steps} steps...")
        engine.run(steps=steps)
        print("\n[Engine] Simulation complete.")
        
        # Save state if simulation completed
        state_file = f"pysrcai/data/logs/ENHANCED_STATE_{timestamp}.json"
        engine.state_container.save_to_file(state_file)
        print(f"\n[Engine] Simulation state saved to: {state_file}")
        
        # Print whiteboard final state
        print("\n[Engine] Final Whiteboard State:")
        for i, entry in enumerate(engine.environment.snapshot()):
            print(f"[{i+1}] {entry}")
        
        print(f"\n[Engine] Full output saved to: {log_filename}")
        
    except Exception as e:
        print(f"\n[Engine] Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Disable async event processing
        event_bus.disable_async()
        
        # Restore original stdout and close log file
        sys.stdout = old_stdout
        tee.close()
        print(f"✅ Log saved to: {log_filename}")


if __name__ == "__main__":
    main()
