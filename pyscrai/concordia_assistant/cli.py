"""Command-line interface for the Concordia Assistant."""

import argparse
import sys
import os
from pathlib import Path
from .rag_pipeline import ConcordiaRAGPipeline


class ConcordiaAssistantCLI:
    """Command-line interface for the Concordia Assistant."""
    
    def __init__(self):
        self.pipeline = None
    
    def run(self):
        """Main entry point for the CLI."""
        parser = self._create_parser()
        args = parser.parse_args()
        
        if hasattr(args, 'func'):
            args.func(args)
        else:
            parser.print_help()
    
    def _create_parser(self):
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description="Concordia Assistant - RAG-powered documentation assistant",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Initialize and ingest documents
  python -m pyscrai.concordia_assistant init
  
  # Start interactive mode
  python -m pyscrai.concordia_assistant chat
  
  # Ask a single question
  python -m pyscrai.concordia_assistant query "How do I create a basic entity in Concordia?"
  
  # Get status information
  python -m pyscrai.concordia_assistant status
            """
        )
        
        parser.add_argument(
            "--config", 
            type=str, 
            help="Path to configuration file (default: config.yaml in the same directory)"
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Init command
        init_parser = subparsers.add_parser("init", help="Initialize the assistant and ingest documents")
        init_parser.add_argument("--force", action="store_true", help="Force re-ingestion of documents")
        init_parser.set_defaults(func=self._init_command)
        
        # Chat command
        chat_parser = subparsers.add_parser("chat", help="Start interactive chat mode")
        chat_parser.set_defaults(func=self._chat_command)
        
        # Query command
        query_parser = subparsers.add_parser("query", help="Ask a single question")
        query_parser.add_argument("question", type=str, help="The question to ask")
        query_parser.add_argument("--no-sources", action="store_true", help="Don't include source information")
        query_parser.set_defaults(func=self._query_command)
        
        # Status command
        status_parser = subparsers.add_parser("status", help="Show status information")
        status_parser.set_defaults(func=self._status_command)
        
        return parser
    
    def _init_command(self, args):
        """Initialize the assistant and ingest documents."""
        print("Initializing Concordia Assistant...")
        
        try:
            self.pipeline = ConcordiaRAGPipeline(args.config)
            self.pipeline.initialize()
            self.pipeline.ingest_documents(force_reingest=args.force)
            print("Initialization completed successfully!")
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            sys.exit(1)
    
    def _chat_command(self, args):
        """Start interactive chat mode."""
        print("Starting Concordia Assistant Chat Mode...")
        print("Type 'exit', 'quit', or 'q' to end the session.")
        print("Type 'help' for more commands.")
        print("=" * 50)
        
        try:
            if not self.pipeline:
                self.pipeline = ConcordiaRAGPipeline(args.config)
                self.pipeline.initialize()
            
            while True:
                try:
                    query = input("\nYou: ").strip()
                    
                    if query.lower() in ['exit', 'quit', 'q']:
                        print("Goodbye!")
                        break
                    elif query.lower() == 'help':
                        self._show_chat_help()
                        continue
                    elif query.lower() == 'status':
                        self._show_status()
                        continue
                    elif query.lower() == 'clear':
                        os.system('cls' if os.name == 'nt' else 'clear')
                        continue
                    elif not query:
                        continue
                    
                    print("\nAssistant: ", end="")
                    response = self.pipeline.query(query)
                    print(response)
                    
                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    break
                except Exception as e:
                    print(f"\nError: {e}")
                    
        except Exception as e:
            print(f"Error starting chat mode: {e}")
            sys.exit(1)
    
    def _query_command(self, args):
        """Handle a single query."""
        try:
            if not self.pipeline:
                self.pipeline = ConcordiaRAGPipeline(args.config)
                self.pipeline.initialize()
            
            response = self.pipeline.query(args.question, include_sources=not args.no_sources)
            print(response)
            
        except Exception as e:
            print(f"Error processing query: {e}")
            sys.exit(1)
    
    def _status_command(self, args):
        """Show status information."""
        try:
            if not self.pipeline:
                self.pipeline = ConcordiaRAGPipeline(args.config)
            
            self._show_status()
            
        except Exception as e:
            print(f"Error getting status: {e}")
            sys.exit(1)
    
    def _show_status(self):
        """Display status information."""
        status = self.pipeline.get_status() if self.pipeline else {"initialized": False}
        
        print("\nConcordia Assistant Status:")
        print("-" * 30)
        print(f"Initialized: {status.get('initialized', False)}")
        print(f"Config Loaded: {status.get('config_loaded', False)}")
        
        if status.get('initialized'):
            print(f"Embedding Provider: {status.get('embedding_provider', 'Unknown')}")
            print(f"Language Model: {status.get('language_model', 'Unknown')}")
            
            vdb_info = status.get('vector_db_info', {})
            if vdb_info:
                print(f"Vector DB Collection: {vdb_info.get('name', 'Unknown')}")
                print(f"Document Count: {vdb_info.get('count', 0)}")
    
    def _show_chat_help(self):
        """Show help for chat mode."""
        print("\nChat Mode Commands:")
        print("- exit, quit, q: Exit the chat")
        print("- help: Show this help message")
        print("- status: Show assistant status")
        print("- clear: Clear the screen")
        print("- Any other text: Ask a question")


def main():
    """Main entry point."""
    cli = ConcordiaAssistantCLI()
    cli.run()


if __name__ == "__main__":
    main()
