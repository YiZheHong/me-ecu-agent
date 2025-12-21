"""
Retrieval functions for querying the vector store.

This module provides the main query interface for retrieving
relevant document chunks based on model constraints.
"""
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from me_ecu_agent.query.meta_store import get_meta_store, MetaStore
from me_ecu_agent.query.doc_selector import select_docs_for_model
from me_ecu_agent.query.config import QueryConfig, get_default_config


def query_chunks_by_model(
    query: str,
    model: str,
    vectorstore: FAISS,
    top_k: int = None,
    meta_store: MetaStore = None,
    config: QueryConfig = None,
) -> List[Tuple[Document, float]]:
    """
    Query the vector store for chunks relevant to a specific model.
    
    This is the primary retrieval function for single-model queries.
    It implements a deterministic retrieval strategy where the model
    identifier is provided explicitly and treated as the sole source
    of truth for document filtering.
    
    Design principles:
    - The model is NOT inferred from the query text.
    - All retrieved chunks must belong to documents applicable to
      the given model (including inherited Base documents for Plus models).
    - This prevents cross-contamination between Base and Plus variants.
    
    Workflow:
    1. Load document metadata from MetaStore (cached).
    2. Select candidate documents based on the model constraint.
    3. Run vector similarity search with an expanded k.
    4. Filter results by allowed doc_uid.
    5. Return top_k results.
    
    Args:
        query: Natural language query string.
        model: Model identifier (e.g., "ECU-750", "ECU-850b").
        vectorstore: FAISS vector store containing document chunks.
        top_k: Number of results to return. If None, uses config default.
        meta_store: MetaStore instance. If None, uses global singleton.
        config: QueryConfig instance. If None, uses default config.
    
    Returns:
        List of (Document, score) tuples, sorted by relevance.
        Returns empty list if no documents match the model.
    
    Example:
        >>> results = query_chunks_by_model(
        ...     "What is the maximum temperature?",
        ...     "ECU-850b",
        ...     vectorstore,
        ... )
        >>> for doc, score in results[:3]:
        ...     print(f"{score:.4f}: {doc.page_content[:100]}")
    """
    # -----------------------------
    # Initialize dependencies
    # -----------------------------
    if config is None:
        config = get_default_config()
    
    if meta_store is None:
        meta_store = get_meta_store(config)
    
    if top_k is None:
        top_k = config.default_top_k
    
    # -----------------------------
    # Step 1: Load metadata
    # -----------------------------
    all_metas = meta_store.get_all()
    
    # -----------------------------
    # Step 2: Select candidate documents
    # -----------------------------
    candidate_docs = select_docs_for_model(model, all_metas)
    allowed_doc_uids = {meta.doc_uid for meta in candidate_docs}
    
    if not allowed_doc_uids:
        # No documents cover this model
        return []
    
    # -----------------------------
    # Step 3: Vector similarity search
    # -----------------------------
    # Use a larger k to avoid recall loss before metadata filtering
    raw_results = vectorstore.similarity_search_with_score(
        query,
        k=config.retrieval_buffer_k,
    )
    
    # -----------------------------
    # Step 4: Filter by doc_uid
    # -----------------------------
    filtered_results = [
        (doc, score)
        for doc, score in raw_results
        if doc.metadata.get("doc_uid") in allowed_doc_uids
    ]
    
    # -----------------------------
    # Step 5: Return top_k
    # -----------------------------
    return filtered_results[:top_k]


def query_chunks_generic(
    query: str,
    vectorstore: FAISS,
    top_k: int = None,
    config: QueryConfig = None,
) -> List[Tuple[Document, float]]:
    """
    Query the vector store without model constraints.
    
    This function is used for general technical questions that don't
    reference a specific model (e.g., "What is CAN bus?").
    
    No metadata filtering is applied - chunks are selected purely
    based on semantic similarity.
    
    Args:
        query: Natural language query string.
        vectorstore: FAISS vector store containing document chunks.
        top_k: Number of results to return. If None, uses config default.
        config: QueryConfig instance. If None, uses default config.
    
    Returns:
        List of (Document, score) tuples, sorted by relevance.
    
    Example:
        >>> results = query_chunks_generic(
        ...     "What is CAN bus used for?",
        ...     vectorstore,
        ... )
    """
    if config is None:
        config = get_default_config()
    
    if top_k is None:
        top_k = config.default_top_k
    
    # Direct similarity search without filtering
    results = vectorstore.similarity_search_with_score(
        query,
        k=top_k,
    )
    
    return results