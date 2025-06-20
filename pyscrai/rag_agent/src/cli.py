"""Command-line interface for RAG Agents."""

import argparse
import os
from typing import List
from .rag_agent_builder import create_agent, quick_agent
from ..config_loader import load_config


class RAGAgentCLI:
    """Command-line interface for interacting with RAG agents."""
    
    def __init__(self):
        self.agent = None
    
    def main(self):
        """Main CLI entry point."""
        parser = argparse.ArgumentParser(description="RAG Agent CLI")
        # --agent-type removed; agent selection is now config-driven
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

        # Config access logging
        parser.add_argument("--log-config-access", action="store_true", help="Log all config variable accesses for debugging")
        
        args = parser.parse_args()
        
        # Set env var for config access logging if enabled
        if args.log_config_access:
            os.environ["RAG_AGENT_LOG_CONFIG_ACCESS"] = "1"
        else:
            os.environ["RAG_AGENT_LOG_CONFIG_ACCESS"] = "0"

        try:
            # Agent selection is now config-driven
            if args.name and args.system_prompt:
                self.agent = quick_agent(
                    name=args.name,
                    system_prompt=args.system_prompt,
                    data_sources=args.data_sources or [],
                    config_file=args.config
                )
            else:
                self.agent = create_agent(
                    config_file=args.config
                )

            # Auto-ingest default data sources from config if present and not already ingested
            if hasattr(self.agent.config, 'agent') and self.agent.config.agent and self.agent.config.agent.data_sources:
                info = self.agent.get_collection_info()
                if info.get('count', 0) == 0:
                    default_paths = [ds['path'] for ds in self.agent.config.agent.data_sources if 'path' in ds]
                    if default_paths:
                        print(f"Auto-ingesting default data sources: {default_paths}")
                        self.agent.ingest_documents(default_paths, force_rebuild=args.force_rebuild)

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
        # Print ASCII logo from ascii_logo.txt (search workspace root and parent dirs)
        possible_paths = [
            os.path.join(os.getcwd(), "ascii_logo.txt"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "ascii_logo.txt"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "ascii_logo.txt")),
        ]
        logo_printed = False
        for logo_path in possible_paths:
            logo_path = os.path.abspath(logo_path)
            if os.path.exists(logo_path):
                try:
                    with open(logo_path, "r", encoding="utf-8") as f:
                        print(f.read())
                    logo_printed = True
                    break
                except Exception:
                    continue
        if not logo_printed:
            print("[Warning] ascii_logo.txt not found. Skipping logo display.")
        print(f"\n {self.agent.get_agent_name()} Interactive Mode")
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
