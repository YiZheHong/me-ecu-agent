"""
Document selection logic for query-time filtering.

This module contains pure functions that determine which documents
should be included in retrieval based on model constraints and
inheritance rules.
"""
from typing import List
from me_ecu_agent.data_schema import DocMeta


def model_is_plus(model: str, metas: List[DocMeta]) -> bool:
    """
    Check whether a given model is a Plus model.
    
    A model is considered Plus if there exists a DocMeta
    that explicitly covers this model and is marked as Plus.
    
    Args:
        model: Model identifier (e.g., "ECU-850b").
        metas: List of all DocMeta objects.
    
    Returns:
        True if the model is Plus, False otherwise.
    """
    for meta in metas:
        if model in meta.covered_models:
            return meta.model_type == "Plus"
    return False


def docs_covering_model(model: str, metas: List[DocMeta]) -> List[DocMeta]:
    """
    Return all documents that directly cover the given model.
    
    This is a hard constraint: only documents explicitly declaring
    coverage of the model are included.
    
    Args:
        model: Model identifier.
        metas: List of all DocMeta objects.
    
    Returns:
        List of DocMeta that cover this model.
    """
    return [
        meta
        for meta in metas
        if model in meta.covered_models
    ]


def base_doc_of_same_series(model: str, metas: List[DocMeta]) -> List[DocMeta]:
    """
    Return the Base document of the same series for a Plus model.
    
    This function implements the inheritance logic:
    if a model is Plus, its Base document is also required
    to provide the full capability baseline.
    
    Args:
        model: Model identifier.
        metas: List of all DocMeta objects.
    
    Returns:
        List containing the Base document of the same series,
        or empty list if the model is not Plus or no Base doc exists.
    """
    # Step 1: Find the series of this Plus model
    series = None
    for meta in metas:
        if model in meta.covered_models and meta.model_type == "Plus":
            series = meta.series
            break
    
    if series is None:
        return []
    
    # Step 2: Return the Base document of the same series
    return [
        meta
        for meta in metas
        if meta.series == series and meta.model_type == "Base"
    ]


def deduplicate_docs(docs: List[DocMeta]) -> List[DocMeta]:
    """
    Remove duplicate documents while preserving one instance per doc_uid.
    
    Deduplication is required because Base documents may be added
    multiple times when multiple models belong to the same series.
    
    Args:
        docs: List of DocMeta (may contain duplicates).
    
    Returns:
        Deduplicated list of DocMeta.
    """
    unique = {}
    for doc in docs:
        unique[doc.doc_uid] = doc
    return list(unique.values())


def select_docs_for_model(model: str, metas: List[DocMeta]) -> List[DocMeta]:
    """
    Select all documents required to answer queries about a specific model.
    
    Selection rules:
    1. Always include documents that directly cover the model.
    2. If the model is Plus, also include the Base document
       of the same series to complete inherited information.
    3. Deduplicate the final document list.
    
    This function is deterministic and model-driven.
    It does not depend on query intent classification.
    
    Args:
        model: Model identifier (e.g., "ECU-750", "ECU-850b").
        metas: List of all DocMeta objects.
    
    Returns:
        List of DocMeta that should be searched for this model.
    
    Example:
        For "ECU-850b" (a Plus model):
        - Includes: ECU-850b Plus doc
        - Includes: ECU-850 Base doc (inheritance)
        
        For "ECU-750" (a Base model):
        - Includes: ECU-750 Base doc only
    """
    docs: List[DocMeta] = []
    
    # Documents explicitly covering this model
    docs += docs_covering_model(model, metas)
    
    return deduplicate_docs(docs)


def select_docs_for_models(models: List[str], metas: List[DocMeta]) -> List[DocMeta]:
    """
    Select documents for multiple models.
    
    This is a convenience wrapper for batch selection.
    Useful for compare queries where multiple models are involved.
    
    Args:
        models: List of model identifiers.
        metas: List of all DocMeta objects.
    
    Returns:
        Deduplicated list of DocMeta covering all specified models.
    """
    docs: List[DocMeta] = []
    
    for model in models:
        docs += select_docs_for_model(model, metas)
    
    return deduplicate_docs(docs)