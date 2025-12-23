"""
Retrieval functions for document fetching.

This module provides synchronous retrieval functions with support for:
- Single model queries
- Multi-model comparison queries
- Generic queries
- Spec-based cross-model comparison
"""
from typing import List, Dict
from me_ecu_agent.query import Retriever


# -----------------------------
# Single model retrieval
# -----------------------------

def retrieve_for_model(
    query: str,
    model: str,
    retriever: Retriever,
    top_k: int = 5,
) -> List[str]:
    """
    Retrieve chunks for a single model.
    
    Args:
        query: User query.
        model: Model identifier (e.g., "ECU-750").
        retriever: Retriever instance from QueryFactory.
        top_k: Number of chunks to retrieve.
    
    Returns:
        List of chunk texts.
    """
    results = retriever.query_by_model(model, query, top_k=top_k)
    chunks = [doc.page_content for doc, _ in results]
    return chunks


# -----------------------------
# Multi-model retrieval
# -----------------------------

def retrieve_for_models(
    query: str,
    models: List[str],
    retriever: Retriever,
    top_k: int = 5,
) -> Dict[str, List[str]]:
    """
    Retrieve chunks for multiple models.
    
    Args:
        query: User query.
        models: List of model identifiers.
        retriever: Retriever instance from QueryFactory.
        top_k: Number of chunks per model.
    
    Returns:
        Dict mapping model -> list of chunk texts.
    
    Example:
        >>> results = retrieve_for_models(
        ...     "What is the max temperature?",
        ...     ["ECU-750", "ECU-850"],
        ...     retriever,
        ... )
        >>> results
        {
            "ECU-750": ["chunk1", "chunk2", ...],
            "ECU-850": ["chunk1", "chunk2", ...],
        }
    """
    results = {}
    
    for model in models:
        chunks = retrieve_for_model(query, model, retriever, top_k)
        results[model] = chunks
    
    return results


# -----------------------------
# Generic retrieval
# -----------------------------

def retrieve_generic(
    query: str,
    retriever: Retriever,
    top_k: int = 10,
) -> List[str]:
    """
    Retrieve chunks for generic queries.
    
    Args:
        query: User query.
        retriever: Retriever instance from QueryFactory.
        top_k: Number of chunks to retrieve (higher for reranking).
    
    Returns:
        List of chunk texts.
    """
    results = retriever.query_generic(query, top_k=top_k)
    chunks = [doc.page_content for doc, _ in results]
    return chunks


# -----------------------------
# Spec-based retrieval for cross-model comparison
# -----------------------------

def retrieve_spec_for_model(
    model: str,
    retriever: Retriever,
    top_k: int = 2,
) -> List[str]:
    """
    Retrieve specification chunks for a single model.
    
    Args:
        model: Model identifier.
        retriever: Retriever instance.
        top_k: Number of spec chunks to retrieve.
    
    Returns:
        List of spec chunk texts.
    """
    results = retriever.query_spec_chunks_by_model(model, top_k=top_k)
    spec_chunks = [doc.page_content for doc, _ in results]
    return spec_chunks


def retrieve_all_model_specs(
    retriever: Retriever,
    top_k: int = 2,
) -> Dict[str, List[str]]:
    """
    Retrieve specification chunks for ALL available models.
    
    This is used for spec-based comparison queries where no specific
    models are mentioned (e.g., "Which model has the highest temperature rating?").
    
    Args:
        retriever: Retriever instance.
        top_k: Number of spec chunks per model.
    
    Returns:
        Dict mapping model -> list of spec chunk texts.
    
    Example:
        >>> results = retrieve_all_model_specs(retriever)
        >>> results
        {
            "ECU-750": ["| Feature | Specification |...", "..."],
            "ECU-850": ["| Feature | Specification |...", "..."],
            "ECU-850b": ["| Feature | Specification |...", "..."],
        }
    """
    # Get all available models
    models = retriever.get_all_models()
    
    # Retrieve spec chunks for each model
    results = {}
    for model in models:
        spec_chunks = retrieve_spec_for_model(model, retriever, top_k)
        
        # Only include models that have spec chunks
        if spec_chunks:
            results[model] = spec_chunks
    
    return results