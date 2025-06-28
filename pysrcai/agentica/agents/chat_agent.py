
"""Chat agent implementation for turn-based conversation."""

from typing import List, Optional, Dict, Any
from ..config.config import AgentConfig
from ..adapters.llm.factory import create_llm
from .base import BaseAgent
import logging
import time


class ChatAgent(BaseAgent):

    def interactive_loop(self):
        """Simple interactive chat loop for CLI."""
        print("Type 'exit' or 'quit' to end the chat.")
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break
                if not user_input:
                    continue
                response = self.chat(user_input)
                print(f"Agent: {response}\n")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    """Agent for engaging in a turn-based chat with the user."""

    def __init__(self, config: AgentConfig):
        """Initialize the Chat agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.logger = logging.getLogger("pysrcai.agentica.agents.chat_agent")
        start = time.time()
        self.logger.info("Initializing LLM...")
        self.llm = create_llm(config.models)
        self.logger.info(f"LLM initialized in {time.time() - start:.2f}s")
        self.conversation_history: List[Dict[str, str]] = []

    def chat(self, user_message: str, **kwargs) -> str:
        """Engage in a turn-based conversation with the user.

        Args:
            user_message: The user's message
            **kwargs: Additional arguments for the LLM

        Returns:
            The agent's response
        """
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})
        # Prepare prompt from history
        prompt = self._build_prompt()
        # Get response from LLM
        try:
            response = self.llm.invoke(prompt, **kwargs)
            if isinstance(response, dict):
                agent_reply = response.get("result", str(response))
            else:
                agent_reply = str(response)
            self.conversation_history.append({"role": "agent", "content": agent_reply})
            return agent_reply
        except Exception as e:
            error_msg = f"Error during chat: {e}"
            self.logger.error(error_msg)
            return error_msg

    def _build_prompt(self) -> str:
        """Build the prompt for the LLM from the conversation history and system prompt."""
        prompt_lines = [self.config.system_prompt.strip() if hasattr(self.config, "system_prompt") else "You are a helpful assistant."]
        for turn in self.conversation_history:
            if turn["role"] == "user":
                prompt_lines.append(f"User: {turn['content']}")
            else:
                prompt_lines.append(f"Agent: {turn['content']}")
        prompt_lines.append("Agent:")
        return "\n".join(prompt_lines)

    def reset_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.conversation_history

    def update_system_prompt(self, new_prompt: str):
        """Update the system prompt for the chat agent."""
        self.config.system_prompt = new_prompt

    def respond(self, message: str) -> str:
        """Generate a response to a given message."""
        # Placeholder implementation
        return f"Echo: {message}"
