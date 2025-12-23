"""
Simple test script for ingest with easy log level switching.

Usage:
    python test_ingest.py           # INFO level (default)
    python test_ingest.py --debug   # DEBUG level
"""
import logging
import sys
from pathlib import Path
from me_ecu_agent.ingest.config import IngestConfig
from me_ecu_agent.ingest.ingest import ingest


# ============================================================
# Simple Logging Setup
# ============================================================

def setup_logging(debug: bool = False):
    """Setup logging - INFO or DEBUG."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )


# ============================================================
# Test Ingest
# ============================================================

def test_ingest(debug: bool = False):
    """Run ingest with specified log level."""
    setup_logging(debug=debug)
    
    # Setup paths
    project_root = Path(__file__).parents[4].resolve()
    data_dir = project_root / "data"
    vector_dir = project_root / "src" / "me_ecu_agent" / "rag"
    meta_dir = project_root / "src" / "me_ecu_agent" / "meta"

    # Create config
    config = IngestConfig.from_dict({
        "project_root": project_root,
        "data_dir": data_dir,
        "vector_dir": vector_dir,
        "meta_dir": meta_dir,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_size": 1500,
        "chunk_overlap": 150,
    })
    
    # Run ingest
    vectorstore = ingest(rebuild=True, config=config)
    
    return vectorstore


# ============================================================
# Main Entry
# ============================================================

if __name__ == "__main__":
    debug = False
    
    print("\n" + "="*80)
    print(f"Running ingest with {'DEBUG' if debug else 'INFO'} logging")
    print("="*80 + "\n")
    
    vectorstore = test_ingest(debug=debug)
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80 + "\n")