"""
QueryFactory Usage Examples and Tests

Simple test script with easy log level switching.

To switch log level, just change the LOG_LEVEL variable below:
    LOG_LEVEL = "INFO"   # Standard output
    LOG_LEVEL = "DEBUG"  # Detailed output
"""

import logging
from pathlib import Path
from typing import List, Tuple
from langchain_core.documents import Document

from me_ecu_agent.query import QueryConfig, QueryFactory


# ============================================================
# CHANGE THIS TO SWITCH LOG LEVEL
# ============================================================
LOG_LEVEL = "INFO"  # Change to "DEBUG" for detailed output
# ============================================================


# Initialize logger
logger = logging.getLogger(__name__)


# ============================================================
# Helper Functions
# ============================================================

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )


def print_results(
    results: List[Tuple[Document, float]],
    title: str = "Query Results",
    max_content_length: int = 200
):
    """
    Pretty print query results with metadata and scores.
    
    Args:
        results: List of (Document, score) tuples
        title: Title for the output
        max_content_length: Max length of content to display
    """
    logger.info("="*80)
    logger.info(title)
    logger.info("="*80)
    logger.info(f"Found {len(results)} results")
    
    for idx, (doc, score) in enumerate(results, 1):
        metadata = doc.metadata
        content = doc.page_content[:max_content_length]
        if len(doc.page_content) > max_content_length:
            content += "..."
        
        logger.info(f"[{idx}] Score: {score:.4f}")
        logger.info(f"    Model: {metadata.get('series', 'N/A')}")
        logger.info(f"    Source: {metadata.get('source_filename', 'N/A')}")
        logger.info(f"    Section: {metadata.get('section_title', 'N/A')}")
        logger.debug(f"    Doc UID: {metadata.get('doc_uid', 'N/A')[:8]}...")
        logger.debug(f"    Chunk Type: {metadata.get('chunk_type', 'N/A')}")
        logger.debug(f"    Content Preview: {content}")


def get_default_config(project_root: Path) -> dict:
    """
    Get default configuration dictionary.
    
    Args:
        project_root: Project root path
        
    Returns:
        Configuration dictionary
    """
    vector_dir = project_root / "src" / "me_ecu_agent" / "rag"
    meta_dir = project_root / "src" / "me_ecu_agent" / "meta"
    
    return {
        "project_root": project_root,
        "vector_dir": vector_dir,
        "meta_dir": meta_dir,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_size": 1500,
        "chunk_overlap": 150,
        "default_top_k": 5,
        "default_threshold_score": 1.5,
        "retrieval_buffer_k": 40,
        "llm_model": "deepseek-chat",
        "llm_temperature": 0.0,
    }


# ============================================================
# Example 1: Generic Query (No Model Constraint)
# ============================================================

def test_query_generic():
    """
    Test generic query without model constraints.
    Searches across all available documentation.
    """
    logger.info("="*80)
    logger.info("TEST 1: Generic Query")
    logger.info("="*80)
    
    project_root = Path(__file__).parents[4].resolve()
    config = get_default_config(project_root)
    
    logger.debug(f"Project root: {project_root}")
    logger.debug(f"Config: {config}")
    
    # Initialize retriever
    logger.info("Initializing retriever...")
    retriever = QueryFactory.from_dict(config)
    
    # Generic query (no model constraint)
    query = "What are CAN bus capabilities?"
    logger.info(f"Query: {query}")
    
    results = retriever.query_generic(query=query)
    
    print_results(results, title="Generic Query Results")
    
    return retriever, results


# ============================================================
# Example 2: Single Model Query
# ============================================================

def test_query_single_model():
    """
    Test querying for a specific ECU model.
    """
    logger.info("="*80)
    logger.info("TEST 2: Single Model Query")
    logger.info("="*80)
    
    project_root = Path(__file__).parents[4].resolve()
    config = get_default_config(project_root)
    
    logger.info("Initializing retriever...")
    retriever = QueryFactory.from_dict(config)
    
    # Query for ECU-750
    model = "ECU-750"
    query = "What is the maximum operating temperature?"
    logger.info(f"Model: {model}")
    logger.info(f"Query: {query}")
    
    results = retriever.query_by_model(
        model=model,
        query=query
    )
    
    print_results(results, title=f"Results for {model}")
    
    return retriever, results


# ============================================================
# Example 3: Model Comparison Query
# ============================================================

def test_query_comparison():
    """
    Test comparison query across multiple models.
    """
    logger.info("="*80)
    logger.info("TEST 3: Model Comparison Query")
    logger.info("="*80)
    
    project_root = Path(__file__).parents[4].resolve()
    config = get_default_config(project_root)
    
    logger.info("Initializing retriever...")
    retriever = QueryFactory.from_dict(config)
    
    # Compare ECU-750 and ECU-850
    models = ["ECU-750", "ECU-850"]
    query = "Compare CAN bus specifications"
    logger.info(f"Models: {', '.join(models)}")
    logger.info(f"Query: {query}")
    
    results = retriever.query_by_models(
        models=models,
        query=query,
        top_k=6  # Get more results for comparison
    )
    
    print_results(results, title="Comparison Query Results")
    
    # Group results by model
    results_by_model = {}
    for doc, score in results:
        model = doc.metadata.get('series', 'Unknown')
        if model not in results_by_model:
            results_by_model[model] = []
        results_by_model[model].append((doc, score))
    
    logger.info("Results grouped by model:")
    for model, model_results in results_by_model.items():
        logger.info(f"  {model}: {len(model_results)} results")
    
    return retriever, results

# ============================================================
# Example 4: Custom Threshold Query
# ============================================================

def test_query_with_threshold():
    """
    Test query with custom score threshold.
    Demonstrates filtering low-quality results.
    """
    logger.info("="*80)
    logger.info("TEST 5: Query with Custom Threshold")
    logger.info("="*80)
    
    project_root = Path(__file__).parents[4].resolve()
    config = get_default_config(project_root)
    
    logger.info("Initializing retriever...")
    retriever = QueryFactory.from_dict(config)
    
    model = "ECU-750"
    query = "What is the maximum operating temperature?"
    
    # Query with default threshold
    logger.info(f"Model: {model}")
    logger.info(f"Query: {query}")
    logger.info(f"Default threshold: {config['default_threshold_score']}")
    
    results_default = retriever.query_by_model(
        model=model,
        query=query
    )
    
    print_results(results_default, title="Results with Default Threshold")
    
    # Query with stricter threshold
    strict_threshold = 1.0
    logger.info(f"Now with stricter threshold: {strict_threshold}")
    
    results_strict = retriever.query_by_model(
        model=model,
        query=query,
        threshold_score=strict_threshold
    )
    
    print_results(results_strict, title="Results with Strict Threshold")
    
    filtered_count = len(results_default) - len(results_strict)
    logger.info(f"Filtered out {filtered_count} results with stricter threshold")
    
    return retriever, results_default, results_strict


# ============================================================
# Example 6: Spec Chunk Query
# ============================================================

def test_query_spec_chunks():
    """
    Test querying specification chunks for a model.
    """
    logger.info("="*80)
    logger.info("TEST 6: Spec Chunk Query")
    logger.info("="*80)
    
    project_root = Path(__file__).parents[4].resolve()
    config = get_default_config(project_root)
    
    logger.info("Initializing retriever...")
    retriever = QueryFactory.from_dict(config)
    
    model = "ECU-750"
    logger.info(f"Model: {model}")
    logger.info("Retrieving specification chunks...")
    
    results = retriever.query_spec_chunks_by_model(
        model=model,
        top_k=3
    )
    
    print_results(results, title=f"Specification Chunks for {model}")
    
    # Verify all are spec chunks
    spec_count = sum(1 for doc, _ in results if doc.metadata.get('chunk_type') == 'spec')
    logger.info(f"Verified: {spec_count}/{len(results)} are spec chunks")
    
    return retriever, results


# ============================================================
# Example 7: All Model Specs Query
# ============================================================

def test_query_all_model_specs():
    """
    Test querying specification chunks for all models.
    """
    logger.info("="*80)
    logger.info("TEST 7: All Model Specs Query")
    logger.info("="*80)
    
    project_root = Path(__file__).parents[4].resolve()
    config = get_default_config(project_root)
    
    logger.info("Initializing retriever...")
    retriever = QueryFactory.from_dict(config)
    
    logger.info("Retrieving specification chunks for all models...")
    
    results = retriever.query_all_model_specs(top_k_per_model=2)
    
    logger.info(f"Retrieved specs for {len(results)} models")
    
    for model, spec_chunks in results.items():
        logger.info(f"\nModel: {model}")
        logger.info(f"  Spec chunks: {len(spec_chunks)}")
        
        for idx, (doc, score) in enumerate(spec_chunks, 1):
            logger.info(f"  [{idx}] Score: {score:.4f}, Section: {doc.metadata.get('section_title')}")
            logger.debug(f"      Content preview: {doc.page_content[:100]}...")
    
    return retriever, results


# ============================================================
# Main Test Runner
# ============================================================

def run_all_tests():
    """
    Run all test examples in sequence.
    """
    logger.info("="*80)
    logger.info("RUNNING ALL QUERY FACTORY TESTS")
    logger.info(f"Log Level: {LOG_LEVEL}")
    logger.info("="*80)
    
    try:
        # Test 1: Generic Query
        logger.info("\n")
        test_query_generic()
        
        # Test 2: Single Model Query
        logger.info("\n")
        test_query_single_model()
        
        # Test 3: Model Comparison
        logger.info("\n")
        test_query_comparison()
        
        # Test 4: Custom Threshold
        logger.info("\n")
        test_query_with_threshold()
        
        # Test 5: Spec Chunk Query
        logger.info("\n")
        test_query_spec_chunks()
        
        # Test 6: All Model Specs
        logger.info("\n")
        test_query_all_model_specs()
        
        logger.info("="*80)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)


def run_single_test(test_name: str):
    """
    Run a single test by name.
    
    Args:
        test_name: Name of the test function (e.g., 'generic', 'single_model', 'comparison')
    """
    test_mapping = {
        'generic': test_query_generic,
        'single_model': test_query_single_model,
        'comparison': test_query_comparison,
        'threshold': test_query_with_threshold,
        'spec_chunks': test_query_spec_chunks,
        'all_specs': test_query_all_model_specs,
    }
    
    if test_name not in test_mapping:
        logger.error(f"Unknown test: {test_name}")
        logger.info(f"Available tests: {', '.join(test_mapping.keys())}")
        return
    
    logger.info(f"Running single test: {test_name}")
    test_mapping[test_name]()


if __name__ == "__main__":
    # Setup logging
    setup_logging(level=LOG_LEVEL)
    
    # Run all tests (default)
    run_all_tests()