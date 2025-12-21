"""
Async retrieval functions for concurrent document fetching.

This module provides async wrappers around the synchronous query functions
to enable concurrent retrieval for comparison queries.
"""
import asyncio
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from me_ecu_agent.query import query_chunks_by_model, query_chunks_generic


async def retrieve_for_model_async(
    query: str,
    model: str,
    vectorstore: FAISS,
    top_k: int = 5,
) -> Tuple[str, List[str]]:
    """
    Asynchronously retrieve chunks for a single model.
    
    This wraps the synchronous query_chunks_by_model in an async executor
    to allow concurrent retrieval.
    
    Args:
        query: User query.
        model: Model identifier (e.g., "ECU-750").
        vectorstore: FAISS vector store.
        top_k: Number of chunks to retrieve.
    
    Returns:
        Tuple of (model, list_of_chunk_texts).
    """
    # Run synchronous function in thread pool
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: query_chunks_by_model(query, model, vectorstore, top_k=top_k)
    )
    
    # Extract text content only
    chunks = [doc.page_content for doc, _ in results]
    
    return model, chunks


async def retrieve_for_models_concurrent(
    query: str,
    models: List[str],
    vectorstore: FAISS,
    top_k: int = 5,
) -> Dict[str, List[str]]:
    """
    Retrieve chunks for multiple models concurrently.
    
    This is the key function for comparison queries - it retrieves
    context for all models in parallel.
    
    Args:
        query: User query.
        models: List of model identifiers.
        vectorstore: FAISS vector store.
        top_k: Number of chunks per model.
    
    Returns:
        Dict mapping model -> list of chunk texts.
    
    Example:
        >>> results = await retrieve_for_models_concurrent(
        ...     "What is the max temperature?",
        ...     ["ECU-750", "ECU-850"],
        ...     vectorstore,
        ... )
        >>> results
        {
            "ECU-750": ["chunk1", "chunk2", ...],
            "ECU-850": ["chunk1", "chunk2", ...],
        }
    """
    # Create tasks for concurrent retrieval
    tasks = [
        retrieve_for_model_async(query, model, vectorstore, top_k)
        for model in models
    ]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Convert list of tuples to dict
    return dict(results)


async def retrieve_generic_async(
    query: str,
    vectorstore: FAISS,
    top_k: int = 10,
) -> List[str]:
    """
    Asynchronously retrieve chunks for generic queries.
    
    Args:
        query: User query.
        vectorstore: FAISS vector store.
        top_k: Number of chunks to retrieve (higher for reranking).
    
    Returns:
        List of chunk texts.
    """
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: query_chunks_generic(query, vectorstore, top_k=top_k)
    )
    
    chunks = [doc.page_content for doc, _ in results]
    
    return chunks


# -----------------------------
# Synchronous wrappers (for backward compatibility)
# -----------------------------

def retrieve_for_models_sync(
    query: str,
    models: List[str],
    vectorstore: FAISS,
    top_k: int = 5,
) -> Dict[str, List[str]]:
    """
    Synchronous wrapper for concurrent retrieval.
    
    Use this when calling from non-async code.
    """
    return asyncio.run(retrieve_for_models_concurrent(query, models, vectorstore, top_k))


def retrieve_generic_sync(
    query: str,
    vectorstore: FAISS,
    top_k: int = 10,
) -> List[str]:
    """
    Synchronous wrapper for generic retrieval.
    """
    return asyncio.run(retrieve_generic_async(query, vectorstore, top_k))