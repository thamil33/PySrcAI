"""Command-line interface for RAG Agents."""

import argparse
import os
from typing import List
from .rag_agent_builder import create_agent, quick_agent
from .config_loader import load_config


class RAGAgentCLI:
    """Command-line interface for interacting with RAG agents."""
    
    def __init__(self):
        self.agent = None
    
    def main(self):
        """Main CLI entry point."""
        parser = argparse.ArgumentParser(description="RAG Agent CLI")
        parser.add_argument("--agent-type", choices=["concordia", "openrouter", "custom"],
                          default="concordia", help="Type of agent to create")
        parser.add_argument("--config", help="Path to configuration file")
        parser.add_argument("--ingest", nargs="+", help="Paths to documents to ingest")
        parser.add_argument("--query", help="Query to ask the agent")
        parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
        parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild of vector database")
        parser.add_argument("--info", action="store_true", help="Show agent and collection info")
        parser.add_argument("--clear", action="store_true", help="Clear the vector database")
        
        # Custom agent options
        parser.add_argument("--name", help="Custom agent name")
        parser.add_argument("--system-prompt", help="Custom system prompt")
        parser.add_argument("--data-sources", nargs="+", help="Data source paths for custom agent")
        
        args = parser.parse_args()
        
        try:
            # Create agent based on type
            if args.agent_type == "custom":
                if not args.name or not args.system_prompt:
                    print("Error: Custom agents require --name and --system-prompt")
                    return
                
                self.agent = quick_agent(
                    name=args.name,
                    system_prompt=args.system_prompt,
                    data_sources=args.data_sources or [],
                    config_file=args.config
                )
            else:
                self.agent = create_agent(
                    agent_type=args.agent_type,
                    config_file=args.config
                )
            
            # Handle commands
            if args.clear:
                self.agent.clear_database()
                print("Vector database cleared.")
                return
            
            if args.ingest:
                print(f"Ingesting documents: {args.ingest}")
                self.agent.ingest_documents(args.ingest, force_rebuild=args.force_rebuild)
            
            if args.info:
                self.show_info()
            
            if args.query:
                response = self.agent.query(args.query)
                print(f"\nResponse:\n{response}")
            
            if args.interactive:
                self.interactive_mode()
        
        except Exception as e:
            print(f"Error: {e}")
    
    def interactive_mode(self):
        """Run in interactive question-answering mode."""
        print(f"\n🤖 {self.agent.get_agent_name()} Interactive Mode")
        print("Type 'quit' to exit, 'info' for collection info, 'clear' to clear database")
        print("-" * 60)
        
        while True:
            try:
                query = input("\n❓ Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if query.lower() == 'info':
                    self.show_info()
                    continue
                
                if query.lower() == 'clear':
                    confirm = input("Are you sure you want to clear the database? (y/N): ")
                    if confirm.lower() == 'y':
                        self.agent.clear_database()
                        print("Database cleared.")
                    continue
                
                if not query:
                    continue
                
                print("\n🔍 Searching for relevant information...")
                response = self.agent.query(query)
                print(f"\n💬 {self.agent.get_agent_name()}:")
                print(response)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def show_info(self):
        """Show agent and collection information."""
        info = self.agent.get_collection_info()
        print("\n📊 Agent Information:")
        print(f"  Agent Name: {self.agent.get_agent_name()}")
        print(f"  Collection: {info['name']}")
        print(f"  Document Count: {info['count']}")
        print(f"  Storage Path: {info['persist_directory']}")


def main():
    """Main entry point for the CLI."""
    cli = RAGAgentCLI()
    cli.main()


if __name__ == "__main__":
    main()
