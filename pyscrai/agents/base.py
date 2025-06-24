"""Base agent interface."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path



# Minimal base agent for all agent types
class BaseAgent(ABC):
    """Minimal abstract base class for all agents (no RAG logic)."""
    pass


# RAG-specific base agent
class BaseRAGAgent(ABC):
    """Abstract base class for RAG agents (retrieval, ingestion, etc)."""

    @abstractmethod
    def ingest(self, doc_paths: List[str], **kwargs) -> List[str]:
        pass

    @abstractmethod
    def query(self, question: str, **kwargs) -> str:
        pass

    @abstractmethod
    def clear_store(self) -> bool:
        pass

    def interactive_loop(self):
        print("Starting interactive mode. Type 'exit' or 'quit' to stop.")
        print("Available commands:")
        print("  - Ask any question to query the agent")
        print("  - 'ingest <path>' to ingest documents")
        print("  - 'clear' to clear the vector store")
        print("  - 'info' to get vector store information")
        print()
        while True:
            try:
                user_input = input(">>> ").strip()
                if user_input.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                if user_input.lower() == 'clear':
                    if self.clear_store():
                        print("Vector store cleared successfully.")
                    else:
                        print("Failed to clear vector store.")
                    continue
                if user_input.lower() == 'info':
                    info = self.get_store_info()
                    print(f"Vector store info: {info}")
                    continue
                if user_input.startswith('ingest '):
                    path = user_input[7:].strip()
                    if Path(path).exists():
                        doc_ids = self.ingest([path])
                        print(f"Ingested {len(doc_ids)} documents from {path}")
                    else:
                        print(f"Path does not exist: {path}")
                    continue
                if not user_input:
                    continue
                response = self.query(user_input)
                print(f"Answer: {response}")
                print()
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def get_store_info(self) -> Dict[str, Any]:
        return {"status": "unknown"}
        return {"status": "unknown"}
