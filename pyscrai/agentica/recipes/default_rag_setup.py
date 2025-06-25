"""
Default RAG Setup Example for PySCRAI

This script demonstrates how to set up a complete RAG system for ingesting
and querying LangChain documentation using PySCRAI's default configuration.

Prerequisites:
1. Set environment variables: OPENROUTER_API_KEY
2. Run: pip install -r requirements.txt
3. Download docs: python recipes/download_langchain_docs.py

Usage:
    python recipes/default_rag_setup.py
"""

import os
import tempfile
import gc
import time
from pathlib import Path
from dotenv import load_dotenv

from pyscrai.config.config import AgentConfig, load_template
from pyscrai.agents.builder import AgentBuilder


def main():
    """Demonstrate default RAG setup with LangChain docs."""
    print("🚀 PySCRAI Default RAG Setup Example")
    print("=" * 50)
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check environment variables
    required_env_vars = ["OPENROUTER_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables and try again.")
        return
    
    print("✅ Environment variables configured")
    
    # Create temporary directory for vector storage
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Load default configuration
        config = load_template("default")
        
        # Override to use local sentence transformers (no API token needed)
        config.embedding.provider = "local_sentencetransformers"
        config.embedding.model = "all-MiniLM-L6-v2"
        config.embedding.device = "cuda"
        
        # Override vector storage to use temporary directory
        config.vectordb.persist_directory = str(Path(temp_dir) / "vectorstore")
        config.vectordb.collection_name = "langgraph_docs"
        
        print(f"📁 Vector storage: {config.vectordb.persist_directory}")
        print(f"🤖 Using local embeddings: {config.embedding.model}")
        
        # Create RAG agent
        print("🔧 Creating RAG agent...")
        agent = AgentBuilder.from_config(config)
        
        # Check if LangChain docs exist
        docs_path = Path("./docs/langchain")
        if not docs_path.exists():
            print("📚 LangChain docs not found. Run: python recipes/download_langchain_docs.py")
            print("Using sample documents instead...")
            
        
        # Ingest documents
        print(f"📥 Ingesting documents from: {docs_path}")
        try:
            doc_ids = agent.ingest([str(docs_path)])
            print(f"✅ Ingested {len(doc_ids)} document chunks")
        except Exception as e:
            print(f"❌ Error during ingestion: {e}")
            return
        
        # Test queries
        test_queries = [
            "What are LangChain chains?",
            "How do embeddings work in LangChain?", 
            "Explain how RAG works",
            "What are the benefits of using RAG?"
        ]
        
        print("\n🔍 Testing RAG queries:")
        print("-" * 30)
        
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            try:
                response = agent.query(query)
                print(f"🤖 Response: {response[:200]}...")
                # Show retrieved context
                print("\n📄 Retrieved context:")
                docs = agent.ingestion_pipeline.vectorstore.similarity_search(query, k=2)
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get('source', 'Unknown')
                    content = doc.page_content[:100]
                    print(f"  {i}. {source}: {content}...")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 30)
        
        # Interactive mode prompt
        print("\n🎯 RAG setup complete!")
        print("To start interactive mode, run:")
        print("  python -m pyscrai.cli --config config/default.yml --interactive")
        
    finally:
        # Proper cleanup to avoid Windows file locking issues
        try:
            # Close the vector store connection
            if 'agent' in locals() and hasattr(agent, 'ingestion_pipeline'):
                if hasattr(agent.ingestion_pipeline, 'vectorstore'):
                    del agent.ingestion_pipeline.vectorstore
            del agent
            gc.collect()
            time.sleep(0.5)  # Give Windows time to release file handles
            
            # Manual cleanup of temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup warning: {cleanup_error}")
            print(f"Temporary files may remain at: {temp_dir}")


if __name__ == "__main__":
    main()