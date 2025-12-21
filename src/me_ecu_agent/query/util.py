from typing import List
from me_ecu_agent.data_schema import DocMeta


def model_is_plus(model: str, metas: List[DocMeta]) -> bool:
    """
    Check whether a given model is a Plus model.

    A model is considered Plus if there exists a DocMeta
    that explicitly covers this model and is marked as Plus.
    """
    for meta in metas:
        if model in meta.covered_models:
            return meta.model_type == "Plus"
    return False


def docs_covering(model: str, metas: List[DocMeta]) -> List[DocMeta]:
    """
    Return all documents that directly cover the given model.

    This is a hard constraint: only documents explicitly declaring
    coverage of the model are included.
    """
    return [
        meta
        for meta in metas
        if model in meta.covered_models
    ]


def base_docs_of_same_series(model: str, metas: List[DocMeta]) -> List[DocMeta]:
    """
    Return the Base document of the same series for a Plus model.

    This function is used to complete inherited information:
    if a model is Plus, its Base document is also required
    to provide the full capability baseline.

    If the model is not Plus or no Base document exists,
    an empty list is returned.
    """
    series = None

    # Locate the Plus document that covers this model
    for meta in metas:
        if model in meta.covered_models and meta.model_type == "Plus":
            series = meta.series
            break

    if series is None:
        return []

    # Return the Base document of the same series
    return [
        meta
        for meta in metas
        if meta.series == series and meta.model_type == "Base"
    ]


def deduplicate(docs: List[DocMeta]) -> List[DocMeta]:
    """
    Remove duplicate documents while preserving one instance per doc_uid.

    Deduplication is required because Base documents may be added
    multiple times when multiple models belong to the same series.
    """
    unique = {}
    for doc in docs:
        unique[doc.doc_uid] = doc
    return list(unique.values())


def select_candidate_docs(
    extracted_models: List[str],
    metas: List[DocMeta],
) -> List[DocMeta]:
    """
    Select candidate documents for retrieval based on extracted models.

    Rules:
    1. Always include documents that directly cover the model.
    2. If the model is Plus, also include the Base document
       of the same series to complete inherited information.
    3. Deduplicate the final document list.

    This function is intentionally deterministic and model-driven.
    It does not depend on query intent classification or comparison logic.
    """
    docs: List[DocMeta] = []

    for model in extracted_models:
        # Documents explicitly covering this model
        docs += docs_covering(model, metas)

        # Inheritance completion: Plus -> Base
        if model_is_plus(model, metas):
            docs += base_docs_of_same_series(model, metas)

    return deduplicate(docs)
