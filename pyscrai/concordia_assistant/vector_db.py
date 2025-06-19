"""ChromaDB adapter for the Concordia Assistant."""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .config_loader import AssistantConfig


class ChromaDBAdapter:
    """Adapter class for ChromaDB vector database operations."""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            persist_dir = self.config.vector_db.persist_directory
            collection_name = self.config.vector_db.collection_name
            
            # Create persistent directory if it doesn't exist
            if not os.path.exists(persist_dir):
                os.makedirs(persist_dir)
            
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(collection_name)
                print(f"Connected to existing collection: {collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "Concordia documentation and guides"}
                )
                print(f"Created new collection: {collection_name}")
            
        except ImportError:
            raise ImportError("chromadb not installed. Run: pip install chromadb")
    
    def add_documents(self, 
                     texts: List[str], 
                     embeddings: List[np.ndarray], 
                     metadatas: List[Dict[str, Any]],
                     ids: Optional[List[str]] = None) -> None:
        """Add documents to the vector database."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        # Convert numpy arrays to lists for ChromaDB
        embedding_lists = [emb.tolist() if isinstance(emb, np.ndarray) else emb 
                          for emb in embeddings]
        
        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embedding_lists,
                metadatas=metadatas
            )
            print(f"Added {len(texts)} documents to the vector database")
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise
    
    def query_similar(self, 
                     query_embedding: np.ndarray, 
                     top_k: int = None,
                     similarity_threshold: float = None) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """Query for similar documents."""
        if top_k is None:
            top_k = self.config.rag.top_k
        if similarity_threshold is None:
            similarity_threshold = self.config.rag.similarity_threshold
        
        # Convert numpy array to list for ChromaDB
        query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        try:
            results = self.collection.query(
                query_embeddings=[query_list],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []
            
            # Convert distances to similarities (ChromaDB returns squared L2 distances)
            # Similarity = 1 / (1 + distance) for a more intuitive similarity score
            similarities = [1 / (1 + dist) for dist in distances]
            
            # Filter by similarity threshold
            filtered_results = []
            for doc, meta, sim in zip(documents, metadatas, similarities):
                if sim >= similarity_threshold:
                    filtered_results.append((doc, meta, sim))
            
            if filtered_results:
                docs, metas, sims = zip(*filtered_results)
                return list(docs), list(metas), list(sims)
            else:
                return [], [], []
                
        except Exception as e:
            print(f"Error querying vector database: {e}")
            return [], [], []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection.name,
                "count": count,
                "metadata": self.collection.metadata
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}
    
    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.config.vector_db.collection_name)
            print(f"Deleted collection: {self.config.vector_db.collection_name}")
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            # ChromaDB doesn't have a direct clear method, so we delete and recreate
            collection_name = self.collection.name
            self.client.delete_collection(collection_name)
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Concordia documentation and guides"}
            )
            print(f"Cleared collection: {collection_name}")
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
