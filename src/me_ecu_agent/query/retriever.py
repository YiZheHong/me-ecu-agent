"""
Retriever: Core retrieval logic without dependency management.

This class focuses purely on retrieval logic, with all dependencies
injected via constructor.
"""
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from me_ecu_agent.query.meta_store import MetaStore
from me_ecu_agent.query.doc_selector import select_docs_for_model, select_docs_for_models
from me_ecu_agent.query.config import QueryConfig


class Retriever:
    """
    Core retrieval functionality.
    
    Handles:
    - Model-specific queries
    - Generic queries
    - Result filtering and ranking
    
    All dependencies are injected in constructor.
    No global state or singleton patterns here.
    """
    
    def __init__(
        self,
        vectorstore: FAISS,
        meta_store: MetaStore,
        config: QueryConfig
    ):
        """
        Initialize Retriever with dependencies.
        
        Args:
            vectorstore: FAISS vector store
            meta_store: MetaStore for document metadata
            config: QueryConfig for parameters
        """
        self.vectorstore = vectorstore
        self.meta_store = meta_store
        self.config = config
    
    def query_by_model(
        self,
        model: str,
        query: str,
        top_k: int = None,
    ) -> List[Tuple[Document, float]]:
        """
        Query for a specific model.
        
        Args:
            model: Model identifier
            query: Query string
            top_k: Number of results. If None, uses config default.
        
        Returns:
            List of (Document, score) tuples
        
        Example:
            >>> results = retriever.query_by_model(
            ...     model="ECU-750",
            ...     query="What is max temperature?"
            ... )
        """
        if top_k is None:
            top_k = self.config.default_top_k
        
        # Get candidate documents for this model
        all_metas = self.meta_store.get_all()
        candidate_docs = select_docs_for_model(model, all_metas)
        allowed_doc_uids = {meta.doc_uid for meta in candidate_docs}
        
        if not allowed_doc_uids:
            return []
        
        # Search with buffer
        raw_results = self.vectorstore.similarity_search_with_score(
            query,
            k=self.config.retrieval_buffer_k,
        )
        
        # Filter by doc_uid
        filtered_results = [
            (doc, score)
            for doc, score in raw_results
            if doc.metadata.get("doc_uid") in allowed_doc_uids
        ]
        
        return filtered_results[:top_k]
    
    def query_by_models(
        self,
        models: List[str],
        query: str,
        top_k: int = None,
    ) -> List[Tuple[Document, float]]:
        """
        Query for multiple models (comparison queries).
        
        Args:
            models: List of model identifiers
            query: Query string
            top_k: Number of results per model
        
        Returns:
            List of (Document, score) tuples
        """
        if top_k is None:
            top_k = self.config.default_top_k
        
        all_metas = self.meta_store.get_all()
        candidate_docs = select_docs_for_models(models, all_metas)
        allowed_doc_uids = {meta.doc_uid for meta in candidate_docs}
        
        if not allowed_doc_uids:
            return []
        
        raw_results = self.vectorstore.similarity_search_with_score(
            query,
            k=self.config.retrieval_buffer_k,
        )
        
        filtered_results = [
            (doc, score)
            for doc, score in raw_results
            if doc.metadata.get("doc_uid") in allowed_doc_uids
        ]
        
        return filtered_results[:top_k]
    
    def query_generic(
        self,
        query: str,
        top_k: int = None,
    ) -> List[Tuple[Document, float]]:
        """
        Generic query without model constraints.
        
        Args:
            query: Query string
            top_k: Number of results
        
        Returns:
            List of (Document, score) tuples
        """
        if top_k is None:
            top_k = self.config.default_top_k
        
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=top_k,
        )
        
        return results