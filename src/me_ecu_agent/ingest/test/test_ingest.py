"""
Configuration and Ingest Usage Examples
"""
from pathlib import Path
from me_ecu_agent.ingest.config import IngestConfig
from me_ecu_agent.ingest.ingest import ingest

# ============================================================
# Example 1: Using IngestConfig object
# ============================================================

def example_ingest_config_object():
    """Use IngestConfig object for more control."""
    print("="*80)
    print("EXAMPLE 1: IngestConfig Object")
    print("="*80)

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
    }
    
    config = IngestConfig.from_dict(agent_config)
    
    # # Print configuration
    print(config)
    print()
    
    # # Use it for ingestion
    vectorstore = ingest(rebuild=True, config=config, verbose=True)
    
    return vectorstore

# ============================================================
# Run examples
# ============================================================

if __name__ == "__main__":    
    example_ingest_config_object()    
    print("Done!")