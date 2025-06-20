"""Vector database adapter for RAG Agents."""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# `AgentConfig` resides in the `src` package; import it explicitly.
from ..config_loader import AgentConfig


class VectorDBAdapter:
    """Adapter class for vector database operations."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = None
        self.collection = None
        # Log config accesses
        self._log_config_access("vector_db.persist_directory", getattr(config.vector_db, 'persist_directory', None))
        self._log_config_access("vector_db.collection_name", getattr(config.vector_db, 'collection_name', None))
        self._initialize_client()

    def _log_config_access(self, key, value):
        try:
            from ..config_access_logger import is_logging_enabled, logger
            if is_logging_enabled():
                logger.info(f"vector_db_adapter accessed {key} -> {repr(value)}")
        except Exception:
            pass

    def _initialize_client(self):
        """Initialize the ChromaDB client."""
        try:
            import chromadb
            from chromadb.config import Settings

            # Create persistent client
            persist_dir = self.config.vector_db.persist_directory
            os.makedirs(persist_dir, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=persist_dir, settings=Settings(anonymized_telemetry=False)
            )

            # Get or create collection
            collection_name = self.config.vector_db.collection_name
            self.collection = self.client.get_or_create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )

            print(f"Initialized ChromaDB with collection: {collection_name}")

        except ImportError:
            raise ImportError("chromadb not installed. Run: pip install chromadb")

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ):
        """Add documents to the vector database."""
        # Convert numpy arrays to lists for ChromaDB
        embedding_lists = [emb.tolist() for emb in embeddings]

        self.collection.add(
            documents=documents,
            embeddings=embedding_lists,
            metadatas=metadatas,
            ids=ids,
        )

    def query(
        self, query_text: str, embedding_adapter, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Query the vector database for similar documents."""
        # Get query embedding
        query_embedding = embedding_adapter.embed_text(query_text)

        # Convert to list for ChromaDB
        query_embedding_list = query_embedding.tolist()

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted_results = []
        if results["documents"] and len(results["documents"]) > 0:
            for i, (doc, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                formatted_results.append(
                    {
                        "content": doc,
                        "metadata": metadata,
                        "similarity": 1 - distance,  # Convert distance to similarity
                        "rank": i + 1,
                    }
                )

        return formatted_results

    def ingest_documents(
        self,
        file_paths: List[str],
        chunker,
        embedding_adapter,
        force_rebuild: bool = False,
    ):
        """Ingest documents into the vector database."""
        if force_rebuild:
            self.clear_collection()

        # Check if collection already has documents
        collection_count = self.collection.count()
        if collection_count > 0 and not force_rebuild:
            print(
                f"Collection already contains {collection_count} documents. Use force_rebuild=True to rebuild."
            )
            return

        print("Processing documents...")
        all_chunks = []
        all_metadatas = []


        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue

            print(f"Processing: {file_path}")

            # Directory
            if os.path.isdir(file_path):
                file_chunks = chunker.chunk_directory(file_path)
            else:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".json":
                    file_chunks = chunker.chunk_json_file(file_path)
                elif ext in [".txt", ".md", ".rst"]:
                    file_chunks = chunker.chunk_text_file(file_path)
                else:
                    print(f"Skipping unsupported file type: {file_path}")
                    continue

            # file_chunks is a list of (chunk, metadata)
            for chunk, metadata in file_chunks:
                all_chunks.append(chunk)
                all_metadatas.append(metadata)

        if not all_chunks:
            print("No documents to process.")
            return

        print(f"Generating embeddings for {len(all_chunks)} chunks...")

        # Generate embeddings in batches
        batch_size = 50  # Process in smaller batches to avoid memory issues

        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i : i + batch_size]
            batch_metadatas = all_metadatas[i : i + batch_size]

            # Generate embeddings for this batch
            batch_embeddings = embedding_adapter.embed_texts(batch_chunks)

            # Create unique IDs for this batch
            batch_ids = [f"chunk_{i + j}" for j in range(len(batch_chunks))]

            # Add to database
            self.add_documents(
                documents=batch_chunks,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )

            print(
                f"Processed batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}"
            )

        final_count = self.collection.count()
        print(
            f"Ingestion complete! Added {final_count} document chunks to the database."
        )

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        return {
            "name": self.collection.name,
            "count": self.collection.count(),
            "persist_directory": self.config.vector_db.persist_directory,
        }

    def clear_collection(self):
        """Clear all documents from the collection."""
        # Delete the collection and recreate it
        collection_name = self.config.vector_db.collection_name
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass  # Collection might not exist

        # Recreate the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        print(f"Cleared collection: {collection_name}")
