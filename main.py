import logging
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from me_ecu_agent.ingest import ingest, IngestConfig
from me_ecu_agent.query import QueryFactory
from me_ecu_agent.agent.graph import build_graph
import io

# ============================================================
# CONFIGURATION
# ============================================================
LOG_LEVEL = "INFO"  # Change to "DEBUG" for detailed output
INPUT_CSV = "test-questions.csv"  # Input file in current directory
OUTPUT_CSV = "test-results.csv"   # Output file in current directory
# ============================================================


# Initialize logger
logger = logging.getLogger(__name__)


# ============================================================
# Logging Setup
# ============================================================

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )

# ============================================================
# Agent Setup
# ============================================================
def get_agent_config(project_root: Path) -> dict:
    """
    Get unified agent configuration for both ingest and query.
    
    Returns:
        agent_config: Single config dict for all operations
    """    
    agent_config = {
        # Paths
        "project_root": project_root,
        "data_dir": project_root / "data",
        "vector_dir": project_root / "rag",
        "meta_dir": project_root / "meta",
        
        # Embedding & Chunking
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_size": 1500,
        "chunk_overlap": 150,
        
        # Query Parameters
        "default_top_k": 3,
        "default_threshold_score": 1.5,
        "retrieval_buffer_k": 40,

        # Agent Query Settings
        "generic_top_k": 5,
        "compare_top_k": 3,
        "single_model_top_k": 3,
        "spec_top_k": 2,
        "max_chars_per_model": 1000,
        "max_chars_per_model_specs": 1000,
    }
    
    logger.debug(f"Agent config created with project root: {project_root}")
    return agent_config

def initialize_agent(agent_config: dict, force_rebuild: bool = False):
    """
    Initialize the complete agent pipeline.
    
    Args:
        agent_config: Unified agent configuration
        force_rebuild: If True, rebuild vector store
    
    Returns:
        app: Compiled LangGraph application
    """
    logger.info("="*80)
    logger.info("AGENT INITIALIZATION")
    logger.info("="*80)
    
    # Step 1: Vector store
    logger.info("Step 1: Initializing vector store...")
    vector_dir = agent_config["vector_dir"]
    faiss_file = vector_dir / "index.faiss"
    pkl_file = vector_dir / "index.pkl"
    
    exists = faiss_file.exists() and pkl_file.exists()
    
    if exists and not force_rebuild:
        logger.info("  Loading existing vector store...")
        vectorstore = ingest(
            rebuild=False,
            config=IngestConfig.from_dict(agent_config),
        )
    else:
        logger.info("  Building new vector store...")
        vectorstore = ingest(
            rebuild=True,
            config=IngestConfig.from_dict(agent_config),
        )
    
    logger.info("  ✓ Vector store ready")
    
    # Step 2: Retriever
    logger.info("Step 2: Initializing retriever...")
    retriever = QueryFactory.from_dict(agent_config)
    logger.info("  ✓ Retriever ready")
    
    # Step 3: Graph
    logger.info("Step 3: Building LangGraph workflow...")
    app = build_graph(retriever, agent_config)
    logger.info("  ✓ Graph ready")
    
    logger.info("="*80)
    logger.info("✓ Agent initialization complete!")
    logger.info("="*80)
    
    return app

def main():
    """Main function to run the agent on test questions."""
    setup_logging(LOG_LEVEL)
    
    logger.info("Starting ECU Agent Testing...")
    
    # Initialize agent
    project_root = Path(__file__).parent.resolve()
    agent_config = get_agent_config(project_root)
    app = initialize_agent(agent_config, force_rebuild=False)

    while True:
        user_query = input("\nEnter your question (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            logger.info("Exiting ECU Agent. Goodbye!")
            break
        start_time = time.time()
        response = app.invoke({"query": user_query})
        logger.info(f"Response:\n{response['answer']}")
        elapsed_time = time.time() - start_time
        logger.info(f"Time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
