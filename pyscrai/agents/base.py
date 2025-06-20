"""Base agent interface."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path


class BaseAgent(ABC):
    """Abstract base class for RAG agents."""
    
    @abstractmethod
    def ingest(self, doc_paths: List[str], **kwargs) -> List[str]:
        """Ingest documents from file paths.
        
        Args:
            doc_paths: List of file or directory paths to ingest
            **kwargs: Additional arguments for ingestion
            
        Returns:
            List of document IDs that were ingested
        """
        pass
    
    @abstractmethod
    def query(self, question: str, **kwargs) -> str:
        """Query the agent with a question.
        
        Args:
            question: The question to ask
            **kwargs: Additional arguments for querying
            
        Returns:
            The agent's response
        """
        pass
    
    @abstractmethod
    def clear_store(self) -> bool:
        """Clear the vector store.
        
        Returns:
            True if successful
        """
        pass
    
    def interactive_loop(self):
        """Start an interactive REPL loop."""
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
                
                # Treat as a query
                response = self.query(user_input)
                print(f"Answer: {response}")
                print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def get_store_info(self) -> Dict[str, Any]:
        """Get information about the vector store.
        
        Returns:
            Dictionary with store information
        """
        # Default implementation - subclasses can override
        return {"status": "unknown"}
