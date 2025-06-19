"""RAG (Retrieval-Augmented Generation) pipeline for the Concordia Assistant."""

import os
from typing import List, Dict, Any, Tuple, Optional
from .config_loader import AgentConfig, load_config
from adapters.embedding_adapter import EmbeddingAdapter
from adapters.vector_db_adapter import VectorDBAdapter
from src.chunking import DocumentChunker
from adapters.llm_adapter import LLMAdapter


class ConcordiaRAGPipeline:
    """Main RAG pipeline for the Concordia Assistant."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.embedding_adapter = None
        self.vector_db = None
        self.chunker = None
        self.llm_adapter = None
        self._initialized = False
    
    def initialize(self):
        """Initialize all components of the RAG pipeline."""
        try:
            print("Initializing Concordia RAG Pipeline...")
            
            # Initialize embedding adapter
            print("Setting up embedding adapter...")
            self.embedding_adapter = EmbeddingAdapter(self.config)
              # Initialize vector database
            print("Setting up vector database...")
            self.vector_db = VectorDBAdapter(self.config)
            
            # Initialize document chunker
            print("Setting up document chunker...")
            self.chunker = DocumentChunker(self.config)
            
            # Initialize language model
            print("Setting up language model...")
            self.llm_adapter = LLMAdapter(self.config)
            
            self._initialized = True
            print("RAG Pipeline initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing RAG pipeline: {e}")
            raise

    def ingest_documents(self, file_paths: List[str], force_reingest: bool = False):
        """Ingest documents from specified file paths."""
        if not self._initialized:
            self.initialize()
        
        # Check if we already have documents in the database
        collection_info = self.vector_db.get_collection_info()
        if collection_info.get("count", 0) > 0 and not force_reingest:
            print(f"Vector database already contains {collection_info['count']} documents. Use force_reingest=True to re-ingest.")
            return
        
        if force_reingest:
            print("Force re-ingesting documents...")
            self.vector_db.clear_collection()
        
        print("Starting document ingestion...")
        all_chunks = []
        for file_path in file_paths:
            print(f"Processing file: {file_path}")
            
            # Check if path exists
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            
            chunks = self._process_file(file_path)
            all_chunks.extend(chunks)
            print(f"Extracted {len(chunks)} chunks from {file_path}")
        
        if not all_chunks:
            print("No documents found to ingest.")
            return
        
        # Generate embeddings and add to vector database
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        self._embed_and_store_chunks(all_chunks)
        
        print(f"Document ingestion completed. Total chunks: {len(all_chunks)}")

    def _process_file(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Process a single file and return chunks."""
        if file_path.endswith('.json'):
            return self.chunker.chunk_json_file(file_path)
        elif os.path.isdir(file_path):
            return self.chunker.chunk_directory(file_path, "text")
        else:
            return self.chunker.chunk_text_file(file_path)
    
    def _embed_and_store_chunks(self, chunks: List[Tuple[str, Dict[str, Any]]]):
        """Generate embeddings for chunks and store in vector database."""
        batch_size = 50  # Process in batches to avoid memory issues
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk[0] for chunk in batch]
            metadatas = [chunk[1] for chunk in batch]
            
            # Generate embeddings
            embeddings = self.embedding_adapter.embed_texts(texts)
            
            # Generate unique IDs
            ids = [f"chunk_{i + j}" for j in range(len(batch))]
            
            # Store in vector database
            self.vector_db.add_documents(texts, embeddings, metadatas, ids)
            
            print(f"Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
    
    def query(self, query_text: str, include_sources: bool = True) -> str:
        """Query the RAG system with a question."""
        if not self._initialized:
            self.initialize()
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_adapter.embed_text(query_text)
            
            # Retrieve relevant documents using the updated method signature
            results = self.vector_db.query(
                query_text,
                self.embedding_adapter,
                top_k=self.config.rag.top_k
            )
            
            if not results:
                return "I couldn't find any relevant information in the documentation to answer your question."
            
            # Extract documents and metadata from results
            documents = [result['content'] for result in results]
            metadatas = [result['metadata'] for result in results]
            similarities = [result['similarity'] for result in results]
            
            # Build context from retrieved documents
            context = self._build_context(documents, metadatas, similarities)
            
            # Generate response using LLM
            response = self._generate_response(query_text, context, include_sources, metadatas)
            
            return response
            
        except Exception as e:
            return f"Error processing query: {e}"
    
    def _build_context(self, documents: List[str], metadatas: List[Dict], similarities: List[float]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        
        for i, (doc, meta, sim) in enumerate(zip(documents, metadatas, similarities)):
            source = meta.get("source", "unknown")
            context_parts.append(f"Document {i+1} (from {os.path.basename(source)}, similarity: {sim:.3f}):\n{doc}\n")
        
        return "\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str, include_sources: bool, metadatas: List[Dict]) -> str:
        """Generate response using the language model."""
        # Build prompt
        prompt = f"""You are a helpful assistant that answers questions about Concordia, a generative social simulation library, and related topics like OpenRouter API usage.

Based on the following context from the documentation, please answer the user's question accurately and concisely.

Context:
{context}

User Question: {query}

Please provide a clear and helpful answer based on the context provided. If the context doesn't contain enough information to fully answer the question, say so and provide what information you can.

Answer:"""
          # Generate response
        response = self.llm_adapter.generate(prompt)
        
        # Add source information if requested
        if include_sources and metadatas:
            sources = set()
            for meta in metadatas:
                source = meta.get("source", "unknown")
                sources.add(os.path.basename(source))
            
            source_text = "\n\nSources: " + ", ".join(sorted(sources))
            response += source_text
        
        return response
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the RAG pipeline."""
        status = {
            "initialized": self._initialized,
            "config_loaded": self.config is not None,
        }
        if self._initialized:
            status.update({
                "embedding_provider": self.config.models.embedding_provider,
                "language_model": self.config.models.language_model,
                "vector_db_info": self.vector_db.get_collection_info() if self.vector_db else {},
            })
        
        return status
