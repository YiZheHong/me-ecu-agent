"""
Retriever: Core retrieval logic without dependency management.

This class focuses purely on retrieval logic, with all dependencies
injected via constructor.
"""
import logging
from typing import List, Tuple, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from me_ecu_agent.query.meta_store import MetaStore
from me_ecu_agent.query.doc_selector import select_docs_for_model, select_docs_for_models
from me_ecu_agent.query.config import QueryConfig

# Initialize logger
logger = logging.getLogger(__name__)


class Retriever:
    """
    Core retrieval functionality.
    
    Handles:
    - Model-specific queries
    - Generic queries
    - Result filtering and ranking
    - Score-based threshold filtering
    - Spec chunk retrieval for cross-model comparison
    
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
            config: QueryConfig for parameters (including threshold_score)
        """
        self.vectorstore = vectorstore
        self.meta_store = meta_store
        self.config = config
        logger.debug(
            f"Retriever initialized with config: "
            f"top_k={config.default_top_k}, "
            f"threshold={config.default_threshold_score}, "
            f"buffer_k={config.retrieval_buffer_k}"
        )
    
    def _filter_by_score(
        self,
        results: List[Tuple[Document, float]],
        threshold: float = None
    ) -> List[Tuple[Document, float]]:
        """
        Filter results by similarity score threshold.
        
        Args:
            results: List of (Document, score) tuples
            threshold: Score threshold. If None, uses config default.
        
        Returns:
            Filtered list of (Document, score) tuples
        
        Note:
            FAISS returns L2 distance scores - lower is better.
            Threshold filters OUT results with score > threshold.
        """
        if threshold is None:
            threshold = self.config.default_threshold_score
        
        initial_count = len(results)
        
        # Filter: keep only results with score <= threshold
        # (lower L2 distance = better similarity)
        filtered = [
            (doc, score)
            for doc, score in results
            if score <= threshold
        ]
        
        filtered_count = len(filtered)
        if filtered_count < initial_count:
            logger.debug(
                f"Score filtering: {initial_count} → {filtered_count} results "
                f"(threshold={threshold:.4f})"
            )
        
        return filtered
    
    def query_by_model(
        self,
        model: str,
        query: str,
        top_k: int = None,
        threshold_score: float = None,
    ) -> List[Tuple[Document, float]]:
        """
        Query for a specific model.
        
        Args:
            model: Model identifier
            query: Query string
            top_k: Number of results. If None, uses config default.
            threshold_score: Score threshold. If None, uses config default.
        
        Returns:
            List of (Document, score) tuples filtered by threshold
        
        Example:
            >>> results = retriever.query_by_model(
            ...     model="ECU-750",
            ...     query="What is max temperature?",
            ...     threshold_score=1.0
            ... )
        """
        if top_k is None:
            top_k = self.config.default_top_k
        
        logger.debug(f"Query by model: model='{model}', query='{query}', top_k={top_k}")
        
        # Get candidate documents for this model
        all_metas = self.meta_store.get_all()
        candidate_docs = select_docs_for_model(model, all_metas)
        allowed_doc_uids = {meta.doc_uid for meta in candidate_docs}
        
        logger.debug(f"Found {len(candidate_docs)} candidate documents for model '{model}'")
        
        if not allowed_doc_uids:
            logger.warning(f"No documents found for model '{model}'")
            return []
        
        # Search with buffer
        raw_results = self.vectorstore.similarity_search_with_score(
            query,
            k=self.config.retrieval_buffer_k,
        )
        logger.debug(f"Retrieved {len(raw_results)} raw results from vectorstore")
        
        # Filter by doc_uid
        filtered_results = [
            (doc, score)
            for doc, score in raw_results
            if doc.metadata.get("doc_uid") in allowed_doc_uids
        ]
        logger.debug(
            f"Filtered by doc_uid: {len(raw_results)} → {len(filtered_results)} results"
        )
        
        # Filter by score threshold
        threshold_filtered = self._filter_by_score(
            filtered_results,
            threshold=threshold_score
        )
        
        final_results = threshold_filtered[:top_k]
        logger.debug(
            f"Query by model complete: returning {len(final_results)} results "
            f"(after threshold and top_k)"
        )
        
        if logger.isEnabledFor(logging.DEBUG):
            for i, (doc, score) in enumerate(final_results[:3]):  # Show first 3
                logger.debug(
                    f"  Result {i+1}: score={score:.4f}, "
                    f"source={doc.metadata.get('source_filename')}, "
                    f"section={doc.metadata.get('section_title')}"
                )
        
        return final_results
    
    def query_by_models(
        self,
        models: List[str],
        query: str,
        top_k: int = None,
        threshold_score: float = None,
    ) -> List[Tuple[Document, float]]:
        """
        Query for multiple models (comparison queries).
        
        Args:
            models: List of model identifiers
            query: Query string
            top_k: Number of results per model
            threshold_score: Score threshold. If None, uses config default.
        
        Returns:
            List of (Document, score) tuples filtered by threshold
        """
        if top_k is None:
            top_k = self.config.default_top_k
        
        logger.debug(
            f"Query by models: models={models}, query='{query}', top_k={top_k}"
        )
        
        all_metas = self.meta_store.get_all()
        candidate_docs = select_docs_for_models(models, all_metas)
        allowed_doc_uids = {meta.doc_uid for meta in candidate_docs}
        
        logger.debug(
            f"Found {len(candidate_docs)} candidate documents for models {models}"
        )
        
        if not allowed_doc_uids:
            logger.warning(f"No documents found for models {models}")
            return []
        
        raw_results = self.vectorstore.similarity_search_with_score(
            query,
            k=self.config.retrieval_buffer_k,
        )
        logger.debug(f"Retrieved {len(raw_results)} raw results from vectorstore")
        
        filtered_results = [
            (doc, score)
            for doc, score in raw_results
            if doc.metadata.get("doc_uid") in allowed_doc_uids
        ]
        logger.debug(
            f"Filtered by doc_uid: {len(raw_results)} → {len(filtered_results)} results"
        )
        
        # Filter by score threshold
        threshold_filtered = self._filter_by_score(
            filtered_results,
            threshold=threshold_score
        )
        
        final_results = threshold_filtered[:top_k]
        logger.debug(
            f"Query by models complete: returning {len(final_results)} results "
            f"(after threshold and top_k)"
        )
        
        if logger.isEnabledFor(logging.DEBUG):
            for i, (doc, score) in enumerate(final_results[:3]):
                logger.debug(
                    f"  Result {i+1}: score={score:.4f}, "
                    f"model={doc.metadata.get('series')}, "
                    f"section={doc.metadata.get('section_title')}"
                )
        
        return final_results
    
    def query_generic(
        self,
        query: str,
        top_k: int = None,
        threshold_score: float = None,
    ) -> List[Tuple[Document, float]]:
        """
        Generic query without model constraints.
        
        Args:
            query: Query string
            top_k: Number of results
            threshold_score: Score threshold. If None, uses config default.
        
        Returns:
            List of (Document, score) tuples filtered by threshold
        """
        if top_k is None:
            top_k = self.config.default_top_k
        
        logger.debug(f"Generic query: query='{query}', top_k={top_k}")
        
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=top_k,
        )
        logger.debug(f"Retrieved {len(results)} results from vectorstore")
        
        # Filter by score threshold
        threshold_filtered = self._filter_by_score(
            results,
            threshold=threshold_score
        )
        
        logger.debug(
            f"Generic query complete: returning {len(threshold_filtered)} results "
            f"(after threshold)"
        )
        
        if logger.isEnabledFor(logging.DEBUG):
            for i, (doc, score) in enumerate(threshold_filtered[:3]):
                logger.debug(
                    f"  Result {i+1}: score={score:.4f}, "
                    f"source={doc.metadata.get('source_filename')}, "
                    f"section={doc.metadata.get('section_title')}"
                )
        
        return threshold_filtered
    
    # ============================================================
    # Spec Chunk Retrieval Methods
    # ============================================================
    
    def get_all_models(self) -> List[str]:
        """
        Get all available ECU models from the metadata store.
        
        Returns:
            List of unique model identifiers sorted alphabetically.
        
        Example:
            >>> retriever.get_all_models()
            ['ECU-750', 'ECU-850', 'ECU-850b']
        """
        all_metas = self.meta_store.get_all()
        models = set()
        
        for meta in all_metas:
            if meta.covered_models:
                models.update(meta.covered_models)
        
        sorted_models = sorted(list(models))
        logger.debug(f"Found {len(sorted_models)} unique models: {sorted_models}")
        
        return sorted_models
    
    def query_spec_chunks_by_model(
        self,
        model: str,
        top_k: int = None,
    ) -> List[Tuple[Document, float]]:
        """
        Query specification chunks for a specific model.
        
        This method specifically targets chunks with chunk_type='spec',
        which contain technical specification tables.
        
        Args:
            model: Model identifier
            top_k: Number of spec chunks to retrieve. If None, uses config default.
        
        Returns:
            List of (Document, score) tuples containing only spec chunks
        
        Example:
            >>> results = retriever.query_spec_chunks_by_model("ECU-750", top_k=2)
            >>> for doc, score in results:
            ...     print(doc.metadata.get("chunk_type"))  # All should be 'spec'
        """
        if top_k is None:
            top_k = self.config.default_top_k
        
        logger.debug(f"Query spec chunks: model='{model}', top_k={top_k}")
        
        # Get candidate documents for this model
        all_metas = self.meta_store.get_all()
        candidate_docs = select_docs_for_model(model, all_metas)
        allowed_doc_uids = {meta.doc_uid for meta in candidate_docs}
        
        logger.debug(f"Found {len(candidate_docs)} candidate documents for model '{model}'")
        
        if not allowed_doc_uids:
            logger.warning(f"No documents found for model '{model}'")
            return []
        
        # Use a generic spec query to find specification chunks
        # We search with a higher buffer to ensure we get enough spec chunks
        raw_results = self.vectorstore.similarity_search_with_score(
            "technical specifications",
            k=self.config.retrieval_buffer_k,
        )
        logger.debug(f"Retrieved {len(raw_results)} raw results from vectorstore")
        
        # Filter by doc_uid and chunk_type='spec'
        spec_results = [
            (doc, score)
            for doc, score in raw_results
            if doc.metadata.get("doc_uid") in allowed_doc_uids
            and doc.metadata.get("chunk_type") == "spec"
        ]
        logger.debug(
            f"Filtered to spec chunks: {len(raw_results)} → {len(spec_results)} results"
        )
        
        # No score threshold for spec chunks - we want all available specs
        # Just return top_k results
        final_results = spec_results[:top_k]
        logger.debug(
            f"Query spec chunks complete: returning {len(final_results)} spec chunks"
        )
        
        if logger.isEnabledFor(logging.DEBUG):
            for i, (doc, score) in enumerate(final_results):
                logger.debug(
                    f"  Spec chunk {i+1}: score={score:.4f}, "
                    f"section={doc.metadata.get('section_title')}"
                )
        
        return final_results
    
    def query_all_model_specs(
        self,
        top_k_per_model: int = None,
    ) -> Dict[str, List[Tuple[Document, float]]]:
        """
        Query specification chunks for ALL available models.
        
        This is used for cross-model spec comparison queries where no specific
        models are mentioned (e.g., "Which model has the highest temperature rating?").
        
        Args:
            top_k_per_model: Number of spec chunks per model. If None, uses config default.
        
        Returns:
            Dict mapping model -> list of (Document, score) tuples containing spec chunks
        
        Example:
            >>> results = retriever.query_all_model_specs(top_k_per_model=2)
            >>> results
            {
                "ECU-750": [(doc1, 0.5), (doc2, 0.6)],
                "ECU-850": [(doc3, 0.4), (doc4, 0.7)],
                "ECU-850b": [(doc5, 0.3), (doc6, 0.8)]
            }
            
            >>> # Access spec content
            >>> for model, spec_chunks in results.items():
            ...     print(f"{model}:")
            ...     for doc, score in spec_chunks:
            ...         print(doc.page_content)  # Full specification table
        """
        if top_k_per_model is None:
            top_k_per_model = self.config.default_top_k
        
        logger.debug(f"Query all model specs: top_k_per_model={top_k_per_model}")
        
        # Get all available models
        models = self.get_all_models()
        logger.debug(f"Querying specs for {len(models)} models: {models}")
        
        # Retrieve spec chunks for each model
        results = {}
        for model in models:
            logger.debug(f"Retrieving specs for model: {model}")
            spec_chunks = self.query_spec_chunks_by_model(model, top_k=top_k_per_model)
            
            # Only include models that have spec chunks
            if spec_chunks:
                results[model] = spec_chunks
                logger.debug(f"  → Found {len(spec_chunks)} spec chunks for {model}")
            else:
                logger.debug(f"  → No spec chunks found for {model}")
        
        logger.debug(
            f"Query all model specs complete: "
            f"retrieved specs for {len(results)}/{len(models)} models"
        )
        
        return results