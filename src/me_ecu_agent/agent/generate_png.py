"""
ECU Agent - LangGraph Workflow Visualization Generator

Generates a PNG image of the LangGraph workflow without running any queries.
Simple, fast, and focused on visualization only.

Usage:
    python generate_graph_png.py
    
Output:
    - langgraph_workflow.png in current directory
"""

import logging
from pathlib import Path
from datetime import datetime

from me_ecu_agent.ingest import ingest, IngestConfig
from me_ecu_agent.query import QueryFactory
from me_ecu_agent.agent.graph import build_graph


# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_PNG = "langgraph_workflow.png"  # Output filename
LOG_LEVEL = "INFO"  # INFO or DEBUG
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
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )


# ============================================================
# Agent Configuration
# ============================================================

def get_agent_config() -> dict:
    """
    Get unified agent configuration.
    
    Returns:
        agent_config: Configuration dictionary
    """
    project_root = Path(__file__).parents[3].resolve()
    
    agent_config = {
        # Paths
        "project_root": project_root,
        "data_dir": project_root / "data",
        "vector_dir": project_root / "src" / "me_ecu_agent" / "rag",
        "meta_dir": project_root / "src" / "me_ecu_agent" / "meta",
        
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


# ============================================================
# Graph Initialization (Minimal)
# ============================================================

def initialize_graph_for_visualization(agent_config: dict):
    """
    Initialize minimal components needed for graph visualization.
    Does NOT rebuild vector store - uses existing one.
    
    Args:
        agent_config: Configuration dictionary
    
    Returns:
        app: Compiled LangGraph application
    """
    logger.info("Initializing components for visualization...")
    
    # Step 1: Load existing vector store (no rebuild)
    logger.info("  Loading vector store...")
    vectorstore = ingest(
        rebuild=False,
        config=IngestConfig.from_dict(agent_config),
    )
    logger.info("    ✓ Vector store loaded")
    
    # Step 2: Initialize retriever
    logger.info("  Initializing retriever...")
    retriever = QueryFactory.from_dict(agent_config)
    logger.info("    ✓ Retriever ready")
    
    # Step 3: Build graph
    logger.info("  Building LangGraph workflow...")
    app = build_graph(retriever, agent_config)
    logger.info("    ✓ Graph built")
    
    return app


# ============================================================
# PNG Generation
# ============================================================

def generate_graph_png(app, output_path: Path) -> bool:
    """
    Generate PNG visualization of the LangGraph workflow.
    
    Args:
        app: Compiled LangGraph application
        output_path: Path to save PNG file
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("Generating graph visualization...")
    
    try:
        # Generate PNG data
        png_data = app.get_graph().draw_mermaid_png()
        
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(png_data)
        
        logger.info(f"  ✓ PNG saved to: {output_path}")
        logger.info(f"  File size: {len(png_data) / 1024:.1f} KB")
        
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Failed to generate PNG: {e}")
        return False


# ============================================================
# Main Function
# ============================================================

def generate_langgraph_png(output_filename: str = OUTPUT_PNG):
    """
    Main function to generate LangGraph workflow PNG.
    
    Args:
        output_filename: Output PNG filename (default: langgraph_workflow.png)
    """
    logger.info("="*80)
    logger.info("ECU AGENT - LANGGRAPH WORKFLOW VISUALIZATION")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    logger.info("")
    
    try:
        # Get output path
        current_dir = Path(__file__).parent
        output_path = current_dir / output_filename
        
        logger.info(f"Output file: {output_path}")
        logger.info("")
        
        # Step 1: Get configuration
        agent_config = get_agent_config()
        
        # Step 2: Initialize graph (minimal, no rebuild)
        app = initialize_graph_for_visualization(agent_config)
        logger.info("")
        
        # Step 3: Generate PNG
        success = generate_graph_png(app, output_path)
        logger.info("")
        
        if success:
            logger.info("="*80)
            logger.info("✅ SUCCESS!")
            logger.info(f"Graph visualization saved to: {output_path.name}")
            logger.info("="*80)
            return output_path
        else:
            logger.error("="*80)
            logger.error("❌ FAILED!")
            logger.error("Could not generate graph visualization")
            logger.error("="*80)
            return None
        
    except FileNotFoundError as e:
        logger.error("")
        logger.error("="*80)
        logger.error("❌ ERROR: Vector store not found!")
        logger.error("="*80)
        logger.error("")
        logger.error("Please run ingestion first:")
        logger.error("  python -m me_ecu_agent.examples.ingest_demo")
        logger.error("")
        return None
        
    except Exception as e:
        logger.error("")
        logger.error("="*80)
        logger.error(f"❌ ERROR: {e}")
        logger.error("="*80)
        logger.error("", exc_info=True)
        return None


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import sys
    
    # Setup logging
    setup_logging(level=LOG_LEVEL)
    
    # Generate PNG
    result = generate_langgraph_png()
    
    if result:
        sys.exit(0)
    else:
        sys.exit(1)