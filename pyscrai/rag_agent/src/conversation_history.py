"""Persistent conversation history for Concordia Assistant (Phase 3)."""
import json
import os
from typing import List, Dict, Any

class ConversationHistory:
    """Stores and retrieves conversation history."""
    def __init__(self, history_path: str = "conversation_history.json"):
        self.history_path = history_path
        self._history = []
        self._load()

    def _load(self):
        if os.path.exists(self.history_path):
            with open(self.history_path, "r", encoding="utf-8") as f:
                self._history = json.load(f)
        else:
            self._history = []

    def add_turn(self, user_query: str, agent_response: str):
        self._history.append({"user": user_query, "agent": agent_response})
        self._save()

    def _save(self):
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2)

    def get_history(self) -> List[Dict[str, str]]:
        return self._history
