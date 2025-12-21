"""
Query module for ECU document retrieval.

This module provides functions to query the vector store
with model-aware and generic retrieval strategies.
"""

from me_ecu_agent.query.retriever import (
    query_chunks_by_model,
    query_chunks_generic,
)
from me_ecu_agent.query.meta_store import (
    get_meta_store,
    reset_meta_store,
)
from me_ecu_agent.query.config import (
    QueryConfig,
    get_default_config,
    set_default_config,
)

__all__ = [
    # Main query functions
    "query_chunks_by_model",
    "query_chunks_generic",
    
    # Meta store management
    "get_meta_store",
    "reset_meta_store",
    
    # Configuration
    "QueryConfig",
    "get_default_config",
    "set_default_config",
]