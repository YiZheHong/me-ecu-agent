"""
ECU Agent Batch Test - CSV Input/Output

Reads test questions from test-questions.csv and outputs results with expected answers.

Input CSV format:
    - question: Test query
    - Expected_Answer: Expected response (for comparison)

Output CSV format:
    - question: Original query
    - expected_answer: Expected response
    - actual_answer: Agent's response
    - intent_type: Detected intent
    - models: Models involved
    - requires_specs: Whether specs were required
    - status: PASS/FAIL (if comparison logic added)
    - execution_time: Time taken (seconds)

To switch log level, change the LOG_LEVEL variable below:
    LOG_LEVEL = "INFO"   # Standard output
    LOG_LEVEL = "DEBUG"  # Detailed output
"""

import logging
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from me_ecu_agent.ingest import ingest, IngestConfig
from me_ecu_agent.query import QueryFactory
from me_ecu_agent.agent.graph import build_graph


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
# Unified Agent Configuration
# ============================================================

def get_agent_config() -> dict:
    """
    Get unified agent configuration for both ingest and query.
    
    Returns:
        agent_config: Single config dict for all operations
    """
    project_root = Path(__file__).parents[4].resolve()
    
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
        "max_chars_per_model": 500,
        "max_chars_per_model_specs": 1000,
    }
    
    logger.debug(f"Agent config created with project root: {project_root}")
    return agent_config


# ============================================================
# CSV Operations
# ============================================================

def read_test_questions(csv_path: Path) -> List[Dict[str, str]]:
    """
    Read test questions from CSV file.
    
    Args:
        csv_path: Path to input CSV file
    
    Returns:
        List of dicts with 'question' and 'Expected_Answer' keys
    """
    logger.info(f"Reading test questions from: {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    
    test_cases = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Validate required columns
        if 'Question' not in reader.fieldnames:
            raise ValueError("CSV must have 'Question' column")
        if 'Expected_Answer' not in reader.fieldnames:
            raise ValueError("CSV must have 'Expected_Answer' column")
        if 'Evaluation_Criteria' not in reader.fieldnames:
            raise ValueError("CSV must have 'Evaluation_Criteria' column")
        
        for row in reader:
            test_cases.append({
                'question': row['Question'].strip(),
                'expected_answer': row['Expected_Answer'].strip(),
                'evaluation_criteria': row['Evaluation_Criteria'].strip()
            })
    
    logger.info(f"Loaded {len(test_cases)} test questions")
    return test_cases


def write_test_results(results: List[Dict[str, Any]], csv_path: Path):
    """
    Write test results to CSV file.
    
    Args:
        results: List of result dictionaries
        csv_path: Path to output CSV file
    """
    logger.info(f"Writing test results to: {csv_path}")
    
    fieldnames = [
        'question',
        'expected_answer',
        'actual_answer',
        'evaluation_criteria',
        'intent_type',
        'models',
        'requires_specs',
        'status',
        'execution_time'
    ]
    
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"✓ Results written successfully ({len(results)} rows)")


# ============================================================
# Agent Initialization
# ============================================================

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


# ============================================================
# Batch Testing
# ============================================================

def run_single_test(app, question: str, expected_answer: str, evaluation_criteria: str) -> Dict[str, Any]:
    """
    Run a single test query.
    
    Args:
        app: LangGraph application
        question: Test question
        expected_answer: Expected answer
        evaluation_criteria: Criteria for evaluating the answer
    
    Returns:
        Result dictionary
    """
    logger.debug(f"Testing: {question[:60]}...")
    
    start_time = time.time()
    
    try:
        result = app.invoke({"query": question})
        execution_time = time.time() - start_time
        
        return {
            'question': question,
            'expected_answer': expected_answer,
            'evaluation_criteria': evaluation_criteria,
            'actual_answer': result.get('answer', 'N/A'),
            'intent_type': result.get('intent', {}).intent_type if hasattr(result.get('intent', {}), 'intent_type') else 'N/A',
            'models': str(result.get('intent', {}).models) if hasattr(result.get('intent', {}), 'models') else 'N/A',
            'requires_specs': str(result.get('intent', {}).requires_specs) if hasattr(result.get('intent', {}), 'requires_specs') else 'N/A',
            'status': 'PASS',
            'execution_time': f"{execution_time:.2f}s"
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"  ✗ Test failed: {e}")
        
        return {
            'question': question,
            'expected_answer': expected_answer,
            'actual_answer': f"ERROR: {str(e)}",
            'intent_type': 'ERROR',
            'models': 'N/A',
            'requires_specs': 'N/A',
            'status': 'FAIL',
            'execution_time': f"{execution_time:.2f}s"
        }


def run_batch_tests(app, test_cases: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Run all test cases.
    
    Args:
        app: LangGraph application
        test_cases: List of test cases with 'question' and 'expected_answer'
    
    Returns:
        List of result dictionaries
    """
    logger.info("="*80)
    logger.info("BATCH TESTING")
    logger.info("="*80)
    logger.info(f"Running {len(test_cases)} test cases...")
    logger.info("")
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"[{i}/{len(test_cases)}] Testing question {i}...")
        
        result = run_single_test(
            app,
            test_case['question'],
            test_case['expected_answer'],
            test_case['evaluation_criteria']
        )
        
        results.append(result)
        
        # Show progress
        if result['status'] == 'PASS':
            logger.info(f"  ✓ Completed in {result['execution_time']}")
        else:
            logger.warning(f"  ✗ Failed in {result['execution_time']}")
        
        logger.debug(f"  Intent: {result['intent_type']}")
        logger.debug(f"  Models: {result['models']}")
        logger.debug("")
    
    return results


# ============================================================
# Summary & Statistics
# ============================================================

def print_summary(results: List[Dict[str, Any]]):
    """Print test summary and statistics."""
    logger.info("="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    total = len(results)
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = total - passed
    
    # Calculate statistics
    execution_times = []
    for r in results:
        try:
            time_str = r['execution_time'].replace('s', '')
            execution_times.append(float(time_str))
        except:
            pass
    
    avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
    total_time = sum(execution_times)
    
    # Intent distribution
    intent_counts = {}
    for r in results:
        intent = r['intent_type']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    logger.info(f"Total Tests: {total}")
    logger.info(f"  Passed: {passed} ({passed/total*100:.1f}%)")
    logger.info(f"  Failed: {failed} ({failed/total*100:.1f}%)")
    logger.info("")
    logger.info("Performance:")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Average time: {avg_time:.2f}s per query")
    logger.info("")
    logger.info("Intent Distribution:")
    for intent, count in sorted(intent_counts.items()):
        logger.info(f"  {intent}: {count} ({count/total*100:.1f}%)")
    
    if failed > 0:
        logger.info("")
        logger.warning("Failed Tests:")
        for i, r in enumerate(results, 1):
            if r['status'] == 'FAIL':
                logger.warning(f"  {i}. {r['question'][:60]}...")
                logger.warning(f"     Error: {r['actual_answer'][:100]}")
    
    logger.info("="*80)
    
    if passed == total:
        logger.info("🎉 All tests passed!")
    else:
        logger.warning(f"⚠️ {failed} test(s) failed")


# ============================================================
# Main Function
# ============================================================

def run_batch_test_demo(
    input_csv: Optional[str] = None,
    output_csv: Optional[str] = None,
    force_rebuild: bool = False
):
    """
    Run batch test demo with CSV input/output.
    
    Args:
        input_csv: Input CSV filename (default: test-questions.csv)
        output_csv: Output CSV filename (default: test-results.csv)
        force_rebuild: If True, rebuild vector store
    """
    logger.info("="*80)
    logger.info("ECU AGENT - BATCH TEST (CSV)")
    logger.info(f"Log Level: {LOG_LEVEL}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    logger.info("")
    
    try:
        # Get file paths
        current_dir = Path(__file__).parent
        input_path = current_dir / (input_csv or INPUT_CSV)
        output_path = current_dir / (output_csv or OUTPUT_CSV)
        
        logger.info(f"Input CSV: {input_path}")
        logger.info(f"Output CSV: {output_path}")
        logger.info("")
        
        # Step 1: Read test questions
        test_cases = read_test_questions(input_path)
        logger.info("")
        
        # Step 2: Initialize agent
        agent_config = get_agent_config()
        app = initialize_agent(agent_config, force_rebuild=force_rebuild)
        logger.info("")
        
        # Step 3: Run tests
        results = run_batch_tests(app, test_cases)
        logger.info("")
        
        # Step 4: Write results
        write_test_results(results, output_path)
        logger.info("")
        
        # Step 5: Print summary
        print_summary(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Batch test failed: {e}", exc_info=True)
        return None


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import sys
    
    # Setup logging
    setup_logging(level=LOG_LEVEL)
    
    # Run batch test
    results = run_batch_test_demo(force_rebuild=False)
    
    if results is not None:
        logger.info("")
        logger.info("✅ Batch test completed successfully!")
    else:
        logger.error("❌ Batch test failed!")
        sys.exit(1)