import re
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

MODEL_PATTERN = re.compile(r"\b[A-Z]{2,4}-\d+[A-Za-z]*\b")


def extract_model_from_query(query: str) -> Optional[str]:
    """
    Extract a model identifier from query string, if present.
    High precision only.
    """
    m = MODEL_PATTERN.search(query)
    return m.group(0) if m else None

def is_chunk_relevant_for_model(
    doc: Document,
    model: str,
) -> bool:
    """
    Decide whether a chunk is relevant for a given model,
    considering Base / Plus inheritance.
    """
    meta = doc.metadata
    covered = meta.get("covered_models", [])
    model_type = meta.get("model_type")
    inherits = meta.get("model_inherits_from")

    # Exact match: Plus chunk for Plus model
    if model in covered:
        return True

    # Inheritance case: Plus model can use Base chunks
    if inherits and model_type == "Base":
        # Example:
        # model = ECU-850b
        # inherits = ECU-800-Base
        # Base chunks are allowed
        return True

    return False

def query_chunks(
    query: str,
    vectorstore: FAISS,
    top_k: int = 8,
) -> List[Document]:
    """
    Query vectorstore and return relevant chunks using metadata-aware filtering.
    """
    # Step 1: Extract model from query (if any)
    model = extract_model_from_query(query)

    # Step 2: Vector similarity search (broad recall)
    candidates = vectorstore.similarity_search(query, k=top_k)

    # Step 3: Metadata filtering
    if model:
        filtered = [
            doc for doc in candidates
            if is_chunk_relevant_for_model(doc, model)
        ]
    else:
        # No model specified → trust similarity only
        filtered = candidates

    # Step 4: Prioritize Plus over Base
    if model:
        filtered.sort(
            key=lambda d: 0 if d.metadata.get("model_type") == "Plus" else 1
        )

    return filtered
