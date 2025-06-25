#!/usr/bin/env python3
"""
Simple PyScRAI Demo
Demonstrates basic RAG functionality with document ingestion and querying.
"""

import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add the parent directory to Python path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from pyscrai.agents.builder import AgentBuilder


def create_sample_documents(temp_dir):
    """Create some sample documents for demonstration."""
    
    # Create a simple text document
    with open(os.path.join(temp_dir, "intro.txt"), "w") as f:
        f.write("""
        Welcome to PyScRAI!
        
        PyScRAI is a Retrieval-Augmented Generation framework that makes it easy
        to build RAG applications. It supports multiple LLM providers, vector stores,
        and document types.
        
        Key features:
        - Vector store integration with ChromaDB
        - Support for OpenRouter and LMStudio LLMs
        - local embeddings
        - Flexible document ingestion pipeline
        - Command-line interface
        """)
    
    # Create a markdown document
    with open(os.path.join(temp_dir, "features.md"), "w") as f:
        f.write("""
        # PyScRAI Features
        
        ## Vector Storage
        PyScRAI uses ChromaDB for persistent vector storage. Documents are
        automatically chunked and embedded for efficient retrieval.
        
        ## LLM Integration
        Multiple LLM providers are supported:
        - OpenRouter API for cloud models
        - LMStudio for local models
        
        ## Configuration
        YAML-based configuration makes it easy to customize:
        - Model parameters
        - Chunking strategies
        - Vector store settings
        """)
    
    # Create a JSON document
    with open(os.path.join(temp_dir, "config_example.json"), "w") as f:
        f.write("""
        {
            "name": "Sample Configuration",
            "description": "This shows how JSON documents can be ingested",
            "settings": {
                "chunk_size": 500,
                "embedding_model": "BAAI/bge-base-en-v1.5",
                "temperature": 0.7
            }
        }
        """)
    
    return temp_dir


def main():
    """Run the PyScRAI demo."""
    
    print("🚀 PyScRAI Demo Starting...")
    print("=" * 50)
    
    # Create temporary documents
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Creating sample documents in: {temp_dir}")
        create_sample_documents(temp_dir)
        
        # Initialize agent with default configuration
        print("\n🤖 Initializing RAG agent...")
        try:
            agent = AgentBuilder.create_default()
            print("✅ Agent created successfully!")
        except Exception as e:
            print(f"❌ Error creating agent: {e}")
            print("\nNote: This demo requires either:")
            print("1. OPENROUTER_API_KEY environment variable for cloud models")
            print("2. Local LMStudio server running for local models")
            return
        
        # Ingest documents
        print(f"\n📚 Ingesting documents from {temp_dir}...")
        try:
            doc_ids = agent.ingest([temp_dir])
            print(f"✅ Ingested {len(doc_ids)} document chunks")
        except Exception as e:
            print(f"❌ Error during ingestion: {e}")
            return
        
        # Get vector store info
        print("\n📊 Vector store information:")
        try:
            info = agent.get_store_info()
            print(f"   Total documents: {info.get('count', 'unknown')}")
            print(f"   Collection: {info.get('collection_name', 'unknown')}")
        except Exception as e:
            print(f"❌ Error getting store info: {e}")
        
        # Demo queries
        demo_queries = [
            "What is PyScRAI?",
            "What LLM providers are supported?",
            "How does the configuration work?",
            "What are the key features?"
        ]
        
        print("\n🔍 Running demo queries...")
        print("-" * 30)
        
        for query in demo_queries:
            print(f"\n❓ Query: {query}")
            try:
                # Get answer with sources
                result = agent.query_with_sources(query)
                
                answer = result.get('answer', 'No answer available')
                sources = result.get('source_documents', [])
                
                print(f"💬 Answer: {answer}")
                
                if sources:
                    print("📄 Sources:")
                    for i, doc in enumerate(sources[:2], 1):  # Show top 2 sources
                        source_file = doc.metadata.get('source', 'unknown')
                        source_name = os.path.basename(source_file)
                        content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                        print(f"   {i}. {source_name}: {content_preview}")
                
            except Exception as e:
                print(f"❌ Error during query: {e}")
        
        # Interactive mode prompt
        print("\n" + "=" * 50)
        print("🎉 Demo completed!")
        print("\nTo try interactive mode, run:")
        print("python -m pyscrai.cli --template default --interactive")
        print("\nOr for more options:")
        print("python -m pyscrai.cli --help")


if __name__ == "__main__":
    main()
