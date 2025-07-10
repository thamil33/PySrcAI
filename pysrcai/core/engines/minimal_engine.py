"""Minimal engine implementation for PySrcAI.

This module provides a minimalist engine implementation for testing core
functionality of the PySrcAI framework. It focuses on a simple debate scenario
with a whiteboard for shared state.
"""

from typing import Dict, List, Optional, Any
from .base_engine import SimulationEngine
from ...agents.environment.minimal_components import MinimalEnvironment, MinimalActionParser, MinimalActionProcessor


class MinimalEngine(SimulationEngine):
    """Minimal implementation of the simulation engine.
    
    This engine implements a simple debate scenario with two agents and a moderator.
    It uses a whiteboard for shared state and focuses on testing core functionality
    like turn scheduling, phase progression, and state management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the minimal engine.
        
        Args:
            config: Configuration dictionary for the simulation.
        """
        super().__init__(config)
        self.environment = MinimalEnvironment([])  # Initialize minimal environment
        self.action_processor = MinimalActionProcessor(self.environment)
        self.phase = "opening"  # Current debate phase
        self.turn_index = 0  # Index within the current phase
        self.debate_topic = config.get("scenario", {}).get("debate_topic", 
                                      "Should we adopt a four-day work week?")
        self.phases = ["opening", "rebuttal", "closing"]
        self.agents_per_phase = 2  # Both proponent and opponent get a turn
        self.word_limit = config.get("engine", {}).get("response_word_limit", 75)
        
    def run(self, steps: Optional[int] = None) -> None:
        """Run the simulation for a specified number of steps.
        
        Args:
            steps: Number of steps to run. If None, uses the value from config.
        """
        if steps is None:
            steps = self.config.get("engine", {}).get("steps", 6)
            
        print(f"[Engine] Running minimal debate simulation for {steps} steps...")
        print(f"[Engine] Debate topic: {self.debate_topic}")
        print(f"[Engine] Initialized with phases: {', '.join(self.phases)}")
        print(f"[Engine] Current phase: {self.phase}")
        print(f"[Engine] Word limit per response: {self.word_limit}")
        
        for _ in range(steps):
            result = self.step()
            if result.get("terminated", False):
                print("[Engine] Simulation terminated.")
                break
                
        print("[Engine] Simulation complete.")
        
    def step(self) -> Dict[str, Any]:
        """Run a single simulation step.
        
        Each step corresponds to one agent's turn in the debate.
        The order is: proponent → opponent → moderator (if phase complete)
        
        Returns:
            Dictionary containing step results.
        """
        self.turn += 1
        print(f"\n[Engine] Step {self.turn}")
        
        # Determine whose turn it is
        agent_index = self.turn_index % len(self.agents)
        current_agent = self.agents[agent_index]
        is_phase_end = (self.turn_index + 1) % self.agents_per_phase == 0
        
        # Get agent action
        action = self._get_agent_action(current_agent)
        self._process_action(current_agent, action)
        
        # Advance turn index
        self.turn_index += 1
        
        # If phase is complete, get moderator assessment and advance phase
        if is_phase_end:
            if self.archon:
                self._get_moderator_assessment()
            
            # Check if we should advance to next phase
            phase_index = self.phases.index(self.phase)
            if phase_index < len(self.phases) - 1:
                self.phase = self.phases[phase_index + 1]
                print(f"[Engine] Advancing to phase: {self.phase}")
            else:
                # Final assessment by moderator
                if self.archon:
                    self._get_moderator_final()
                return {"terminated": True}
                
        return {"terminated": False}
        
    def _get_agent_action(self, agent: Any) -> Dict[str, Any]:
        """Get an action from an agent.
        
        Args:
            agent: The agent to get an action from.
            
        Returns:
            Dictionary containing the action.
        """
        # Build prompt for agent
        prompt = self._build_agent_prompt(agent)
        
        # Create action spec for the agent
        from ...agents.base.agent import ActionSpec, OutputType
        action_spec = ActionSpec(
            call_to_action=prompt,
            output_type=OutputType.FREE,
            tag='debate_response'
        )
        
        # Get response from agent using the proper interface
        response = agent.act(action_spec)
        
        # Parse action from response
        action = self._parse_agent_response(response)
        
        return action
        
    def _build_agent_prompt(self, agent: Any) -> str:
        """Build a prompt for an agent.
        
        Args:
            agent: The agent to build a prompt for.
            
        Returns:
            Prompt string.
        """
        agent_role = "Proponent" if agent.name == self.agents[0].name else "Opponent"
        prompt_parts = [
            f"You are the {agent_role} in a debate on the topic: '{self.debate_topic}'",
            f"Current phase: {self.phase}",
            "Whiteboard contents:",
        ]
        
        for i, entry in enumerate(self.environment.snapshot()):
            prompt_parts.append(f"[{i+1}] {entry}")
            
        if self.phase == "opening":
            prompt_parts.append("\nPresent your opening argument in favor of your position.")
            action_template = "ACTION: propose"
        elif self.phase == "rebuttal":
            prompt_parts.append("\nRespond to your opponent's arguments.")
            action_template = "ACTION: respond"
        elif self.phase == "closing":
            prompt_parts.append("\nPresent your closing summary.")
            action_template = "ACTION: respond"
            
        prompt_parts.append(f"\nRespond using this format:")
        prompt_parts.append(f"{action_template}")
        prompt_parts.append(f"CONTENT: <your argument in {self.word_limit} words or less>")
        
        return "\n".join(prompt_parts)
        
    def _parse_agent_response(self, response: str) -> Dict[str, Any]:
        """Parse an agent's response into an action using the MinimalActionParser.
        
        Args:
            response: The agent's response string.
            
        Returns:
            Dictionary containing the parsed action.
        """
        try:
            # Use MinimalActionParser to enforce the strict format
            verb, content, exceeded_limit = MinimalActionParser.parse(response)
            
            result = {
                "type": verb,
                "content": content,
                "valid": True
            }
            
            # Add warning if word limit was exceeded
            if exceeded_limit:
                word_count = len(content.split())
                result["warning"] = f"Response exceeded {self.word_limit}-word limit (got {word_count} words)."
                print(f"[Engine] Warning: {result['warning']}")
            
            return result
        except ValueError as e:
            print(f"[Engine] Invalid response format: {e}")
            # Return a fallback action if parsing fails
            return {
                "type": "invalid",
                "content": "Invalid response format. Please follow the template exactly.",
                "valid": False,
                "error": str(e)
            }
        
    def _process_action(self, agent: Any, action: Dict[str, Any]) -> None:
        """Process an agent's action.
        
        Args:
            agent: The agent who performed the action.
            action: The action to process.
        """
        agent_name = agent.name
        action_type = action.get("type", "")
        content = action.get("content", "")
        
        print(f"{agent_name} acts: {action_type}")
        print(content)
        
        # Add to environment whiteboard
        self.environment.add_note(f"{agent_name}: {content}")
        
    def _get_moderator_assessment(self) -> None:
        """Get an assessment from the moderator.
        
        This is called at the end of each phase.
        """
        if not self.archon:
            return
            
        prompt = self._build_moderator_prompt()
        
        # Create action spec for the moderator
        from ...agents.base.agent import ActionSpec, OutputType
        action_spec = ActionSpec(
            call_to_action=prompt,
            output_type=OutputType.FREE,
            tag='moderator_assessment'
        )
        
        response = self.archon.act(action_spec)
        assessment = self._parse_moderator_response(response)
        
        print(f"[Moderator] {assessment.get('content', '')}")
        self.environment.add_note(f"Moderator: {assessment.get('content', '')}")
        
    def _build_moderator_prompt(self) -> str:
        """Build a prompt for the moderator.
        
        Returns:
            Prompt string.
        """
        prompt_parts = [
            "You are the Moderator in a debate.",
            f"Topic: '{self.debate_topic}'",
            f"Phase just completed: {self.phase}",
            "Whiteboard contents:",
        ]
        
        for i, entry in enumerate(self.environment.snapshot()):
            prompt_parts.append(f"[{i+1}] {entry}")
            
        prompt_parts.append("\nProvide a brief assessment of this phase of the debate.")
        prompt_parts.append("ACTION: respond")
        prompt_parts.append(f"CONTENT: <your assessment in {self.word_limit} words or less>")
        
        return "\n".join(prompt_parts)
        
    def _parse_moderator_response(self, response: str) -> Dict[str, Any]:
        """Parse the moderator's response.
        
        Args:
            response: The moderator's response string.
            
        Returns:
            Dictionary containing the parsed response.
        """
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
        
    def _get_moderator_final(self) -> None:
        """Get the final judgment from the moderator.
        
        This is called at the end of the simulation.
        """
        if not self.archon:
            return
            
        prompt = self._build_final_prompt()
        
        # Create action spec for the moderator
        from ...agents.base.agent import ActionSpec, OutputType
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
        
    def _build_final_prompt(self) -> str:
        """Build the final prompt for the moderator.
        
        Returns:
            Prompt string.
        """
        prompt_parts = [
            "You are the Moderator in a debate that has just concluded.",
            f"Topic: '{self.debate_topic}'",
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