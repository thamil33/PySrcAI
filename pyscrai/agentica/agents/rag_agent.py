"""RAG agent implementation using LangChain."""

from typing import List, Optional, Dict, Any
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


from ..config.config import AgentConfig
from ..adapters.llm.factory import create_llm
from ..ingestion.pipeline import IngestionPipeline
from .base import BaseRAGAgent
import logging
import time


class RAGAgent(BaseRAGAgent):
    """RAG agent implementation leveraging LangChain's retrieval QA chain."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the RAG agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.logger = logging.getLogger("pyscrai.agentica.agents.rag_agent")
        start = time.time()
        self.logger.info("Initializing LLM...")
        self.llm = create_llm(config.models)
        self.logger.info(f"LLM initialized in {time.time() - start:.2f}s")

        start = time.time()
        self.logger.info("Initializing ingestion pipeline...")
        self.ingestion_pipeline = IngestionPipeline(config)
        self.logger.info(f"Ingestion pipeline initialized in {time.time() - start:.2f}s")

        start = time.time()
        self.logger.info("Setting up QA chain...")
        self._setup_qa_chain()
        self.logger.info(f"QA chain setup in {time.time() - start:.2f}s")
    
    def _setup_qa_chain(self):
        """Set up the retrieval QA chain."""
        # Create custom prompt template
        prompt_template = PromptTemplate(
            template=self._get_prompt_template(),
            input_variables=["context", "question"]
        )
        
        # Create retriever from vector store
        retriever = self.ingestion_pipeline.vectorstore.vectorstore.as_retriever(
            search_kwargs={
                "k": self.config.rag.top_k,
                # Add any additional search parameters
            }
        )
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Could be configurable
            retriever=retriever,
            chain_type_kwargs={
                "prompt": prompt_template,
                "verbose": True  # Could be configurable
            },
            return_source_documents=True
        )
    
    def _get_prompt_template(self) -> str:
        """Get the prompt template for the QA chain.
        
        Returns:
            Prompt template string
        """
        return f"""{self.config.system_prompt}

Context information:
{{context}}

Question: {{question}}

Answer the question based on the context information provided above. If the context doesn't contain enough information to answer the question, say so clearly.

Answer:"""
    
    def ingest(self, doc_paths: List[str], **kwargs) -> List[str]:
        """Ingest documents from file paths.
        
        Args:
            doc_paths: List of file or directory paths to ingest
            **kwargs: Additional arguments for ingestion
            
        Returns:
            List of document IDs that were ingested
        """
        all_doc_ids = []
        
        for path in doc_paths:
            try:
                from pathlib import Path
                path_obj = Path(path)
                
                if path_obj.is_file():
                    doc_ids = self.ingestion_pipeline.ingest_files([path], **kwargs)
                elif path_obj.is_dir():
                    doc_ids = self.ingestion_pipeline.ingest_directory(path, **kwargs)
                else:
                    print(f"Warning: Path does not exist: {path}")
                    continue
                
                all_doc_ids.extend(doc_ids)
                print(f"Ingested {len(doc_ids)} documents from {path}")
                
            except Exception as e:
                print(f"Error ingesting {path}: {e}")
        
        return all_doc_ids
    
    def query(self, question: str, **kwargs) -> str:
        """Query the agent with a question.
        
        Args:
            question: The question to ask
            **kwargs: Additional arguments for querying
            
        Returns:
            The agent's response
        """
        try:
            # Use the QA chain to get response
            result = self.qa_chain.invoke({"query": question})
            
            # Extract the answer
            if isinstance(result, dict):
                return result.get("result", "No answer found.")
            else:
                return str(result)
                
        except Exception as e:
            return f"Error processing query: {e}"
    
    def query_with_sources(self, question: str, **kwargs) -> Dict[str, Any]:
        """Query the agent and return both answer and source documents.
        
        Args:
            question: The question to ask
            **kwargs: Additional arguments for querying
            
        Returns:
            Dictionary with answer and source documents
        """
        try:
            result = self.qa_chain.invoke({"query": question})
            
            if isinstance(result, dict):
                return {
                    "answer": result.get("result", "No answer found."),
                    "source_documents": result.get("source_documents", [])
                }
            else:
                return {
                    "answer": str(result),
                    "source_documents": []
                }
                
        except Exception as e:
            return {
                "answer": f"Error processing query: {e}",
                "source_documents": []
            }
    
    def clear_store(self) -> bool:
        """Clear the vector store.
        
        Returns:
            True if successful
        """
        return self.ingestion_pipeline.clear_vectorstore()
    
    def get_store_info(self) -> Dict[str, Any]:
        """Get information about the vector store.
        
        Returns:
            Dictionary with store information
        """
        return self.ingestion_pipeline.get_vectorstore_info()
    
    def search_documents(
        self, 
        query: str, 
        k: Optional[int] = None,
        with_scores: bool = False,
        **kwargs
    ) -> List[Any]:
        """Search for relevant documents without generating an answer.
        
        Args:
            query: Search query
            k: Number of documents to return
            with_scores: Whether to include similarity scores
            **kwargs: Additional search arguments
            
        Returns:
            List of documents or (document, score) tuples
        """
        if with_scores:
            return self.ingestion_pipeline.search_with_scores(query, k, **kwargs)
        else:
            return self.ingestion_pipeline.search(query, k, **kwargs)
    
    def update_system_prompt(self, new_prompt: str):
        """Update the system prompt and recreate the QA chain.
        
        Args:
            new_prompt: New system prompt
        """
        self.config.system_prompt = new_prompt
        self._setup_qa_chain()
    
    def add_preprocessing_hook(self, hook):
        """Add a preprocessing hook to the ingestion pipeline.
        
        Args:
            hook: Function that takes a Document and returns a modified Document
        """
        self.ingestion_pipeline.add_preprocessing_hook(hook)
