"""Enhanced CLI for   PyScRAI with full functionality."""


import argparse
from json import load
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import logging
import time
load_dotenv()



# Update import to use correct package structure
from ..config.config import load_config, load_template, list_templates
from ..agents.builder import AgentBuilder, create_agent


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="  PyScRAI - Retrieval Augmented Generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cli --config my_config.yaml --ingest docs/
  python -m cli --template default --query "What is RAG?"
  python -m cli --template local_models --interactive
  python -m cli --config my_config.yaml --clear
        """
    )
    
    # Configuration options
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--config", 
        type=str, 
        help="Custom config YAML file"
    )
    config_group.add_argument(
        "--template", 
        type=str, 
        choices=list_templates(), 
        default="default",
        help="Use a template configuration (default: default)"
    )
    
    # Action options  
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--ingest", 
        type=str, 
        help="Ingest documents from a file or directory path"
    )
    action_group.add_argument(
        "--query", 
        type=str, 
        help="Ask a question and get an answer"
    )
    action_group.add_argument(
        "--interactive", 
        action="store_true",
        help="Start interactive REPL mode"
    )
    action_group.add_argument(
        "--clear", 
        action="store_true",
        help="Clear the vector store"
    )
    action_group.add_argument(
        "--info", 
        action="store_true",
        help="Show vector store information"
    )
    
    # Additional options
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("pyscrai.cli")

    # If no action specified, default to info
    if not any([args.ingest, args.query, args.interactive, args.clear, args.info]):
        args.info = True

    import yaml
    def get_agent_type_from_yaml(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            # Try both top-level and agent section
            if isinstance(data, dict):
                if 'agent_type' in data:
                    return data['agent_type']
                if 'agent' in data and isinstance(data['agent'], dict):
                    return data['agent'].get('agent_type', 'rag')
        except Exception:
            pass
        return 'rag'

    try:
        # Create agent
        start = time.time()
        if args.verbose:
            logger.info("Loading configuration...")

        agent_type = 'rag'
        config_source = None
        if args.config:
            agent_type = get_agent_type_from_yaml(args.config)
            agent = AgentBuilder.from_config_file(args.config, agent_type=agent_type)
            config_source = f"config file: {args.config}"
        else:
            # Find template YAML path
            from ..config.templates import get_template_path
            template_path = get_template_path(args.template)
            agent_type = get_agent_type_from_yaml(template_path)
            agent = AgentBuilder.from_template(args.template, agent_type=agent_type)
            config_source = f"template: {args.template}"

        logger.info(f"Agent created using {config_source} in {time.time() - start:.2f}s")
        if args.verbose:
            # Only print vectorstore info if present
            vectordb = getattr(agent.config, 'vectordb', None)
            if vectordb and hasattr(vectordb, 'collection_name'):
                logger.info(f"Collection: {vectordb.collection_name}")
            logger.info(f"LLM: {agent.config.models.provider}/{agent.config.models.model}")
            embedding = getattr(agent.config, 'embedding', None)
            if embedding and hasattr(embedding, 'provider'):
                logger.info(f"Embeddings: {embedding.provider}/{embedding.model}")

        # Execute action
        if args.ingest:
            ingest_documents(agent, args.ingest, args.verbose)
        elif args.query:
            query_agent(agent, args.query, args.verbose)
        elif args.interactive:
            interactive_mode(agent)
        elif args.clear:
            clear_vectorstore(agent, args.verbose)
        elif args.info:
            show_info(agent, args.verbose)

    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def ingest_documents(agent, path: str, verbose: bool = False):
    """Ingest documents from a path."""
    if verbose:
        print(f"Ingesting documents from: {path}")
    
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)
    
    try:
        doc_ids = agent.ingest([path])
        print(f"Successfully ingested {len(doc_ids)} documents.")
        
        if verbose:
            info = agent.get_store_info()
            print(f"Vector store now contains {info.get('count', 'unknown')} documents.")
            
    except Exception as e:
        print(f"Error during ingestion: {e}")
        sys.exit(1)


def query_agent(agent, question: str, verbose: bool = False):
    """Query the agent with a question."""
    if verbose:
        print(f"Querying: {question}")
        print()
    
    try:
        # Get answer with sources for verbose mode
        if verbose:
            result = agent.query_with_sources(question)
            print("Answer:")
            print(result["answer"])
            print()
            
            if result["source_documents"]:
                print("Sources:")
                for i, doc in enumerate(result["source_documents"][:3], 1):
                    source = doc.metadata.get("source", "Unknown")
                    print(f"{i}. {source}")
            else:
                print("No source documents found.")
        else:
            answer = agent.query(question)
            print(answer)
            
    except Exception as e:
        print(f"Error during query: {e}")
        sys.exit(1)


def interactive_mode(agent):
    """Start interactive REPL mode."""
    print("Starting interactive mode...")
    print(f"Collection: {agent.config.vectordb.collection_name}")
    print("[DEBUG] Entering interactive_loop()...")
    agent.interactive_loop()
    print("[DEBUG] Exited interactive_loop()")


def clear_vectorstore(agent, verbose: bool = False):
    """Clear the vector store."""
    if verbose:
        info = agent.get_store_info()
        current_count = info.get('count', 'unknown')
        print(f"Current vector store contains {current_count} documents.")
    
    try:
        success = agent.clear_store()
        if success:
            print("Vector store cleared successfully.")
        else:
            print("Failed to clear vector store.")
            sys.exit(1)
    except Exception as e:
        print(f"Error clearing vector store: {e}")
        sys.exit(1)


def show_info(agent, verbose: bool = False):
    """Show vector store information."""
    try:
        info = agent.get_store_info()
        
        print("Vector Store Information:")
        print("=" * 30)
        print(f"Collection: {info.get('name', 'Unknown')}")
        print(f"Document count: {info.get('count', 'Unknown')}")
        print(f"Persist directory: {info.get('persist_directory', 'Unknown')}")
        print(f"Embedding function: {info.get('embedding_function', 'Unknown')}")
        
        if verbose:
            print("\nConfiguration:")
            print(f"LLM Provider: {agent.config.models.provider}")
            print(f"LLM Model: {agent.config.models.model}")
            print(f"Embedding Provider: {agent.config.embedding.provider}")  
            print(f"Embedding Model: {agent.config.embedding.model}")
            print(f"Chunk Size: {agent.config.chunking.chunk_size}")
            print(f"Top K: {agent.config.rag.top_k}")
            
            if info.get('error'):
                print(f"\nError: {info['error']}")
                
    except Exception as e:
        print(f"Error getting info: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
