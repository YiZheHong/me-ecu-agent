"""
QueryFactory Usage Examples

This file demonstrates all the ways to use QueryFactory to initialize
the query system for different scenarios.
"""

from pathlib import Path
from me_ecu_agent.query import QueryConfig, QueryFactory

# ============================================================
# Example 1: Using dictionary configuration
# ============================================================

def example_from_dict():
    """
    Load configuration from a dictionary.
    Useful for programmatic configuration from environment variables or JSON.
    """
    project_root = Path(__file__).parents[4].resolve()
    data_dir = project_root / "data"
    vector_dir= project_root / "src" / "me_ecu_agent" / "rag"
    meta_dir = project_root / "src" / "me_ecu_agent" / "meta"

    # Create config from dict

    agent_config = {
        "project_root": project_root,
        "data_dir": data_dir,
        "vector_dir": vector_dir,
        "meta_dir": meta_dir,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_size": 1500,
        "chunk_overlap": 150,
        "default_top_k": 5,
        "retrieval_buffer_k": 40,
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": "deepseek-chat",
        "llm_temperature": 0.0,
    }
    
    retriever = QueryFactory.from_dict(agent_config)
    
    # Generic query (no model constraint)
    results = retriever.query_generic(
        query="What are CAN bus capabilities?"
    )

    return retriever, results
    
if __name__ == "__main__":
    project_root = Path(__file__).parents[4].resolve()
    example_from_dict()
