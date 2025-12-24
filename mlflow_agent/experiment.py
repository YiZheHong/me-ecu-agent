"""
ECU Agent MLflow Experiment

Run experiments with MLflow tracking and model registration.
All artifacts (vector store, metadata, configs) are saved to the run's artifact directory.

Usage:
    python experiment.py
    python experiment.py --rebuild  # Force rebuild vector store
    python experiment.py --experiment-name "my_experiment"
"""

import logging
import mlflow
import mlflow.pyfunc
import argparse
import time
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

from mlflow_model import ECUAgentWrapper
from mlflow_utils import (
    get_agent_config,
    initialize_vector_store,
    initialize_retriever,
    log_model_to_mlflow,
    log_metrics,
    log_params,
    setup_mlflow_experiment,
    get_artifact_path_for_run,
    get_timestamp,
    format_duration,
)
from me_ecu_agent.agent.graph import build_graph


# ============================================================
# CONFIGURATION
# ============================================================
LOG_LEVEL = "INFO"  # Change to "DEBUG" for detailed output
DEFAULT_EXPERIMENT_NAME = "ecu_agent_experiments"
DEFAULT_TEST_CSV = "test-questions.csv"
# ============================================================


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
# Test Data Loading
# ============================================================

def load_test_questions(csv_path: Path) -> List[Dict[str, str]]:
    """
    Load test questions from CSV file.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        List of test cases with query, expected_answer, evaluation_criteria
    """
    logger.info(f"Loading test questions from: {csv_path}")
    
    if not csv_path.exists():
        logger.warning(f"Test CSV not found: {csv_path}")
        return []
    
    test_cases = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Check if required columns exist
        if 'Question' not in reader.fieldnames:
            logger.warning("CSV missing 'Question' column, skipping test data")
            return []
        
        for row in reader:
            test_cases.append({
                'query': row.get('Question', '').strip(),
                'expected_answer': row.get('Expected_Answer', '').strip(),
                'evaluation_criteria': row.get('Evaluation_Criteria', '').strip()
            })
    
    logger.info(f"Loaded {len(test_cases)} test questions")
    return test_cases


# ============================================================
# Agent Initialization
# ============================================================

def initialize_agent(agent_config: Dict[str, Any], force_rebuild: bool = False):
    """
    Initialize the complete agent pipeline.
    
    Args:
        agent_config: Unified agent configuration
        force_rebuild: If True, rebuild vector store
    
    Returns:
        app: Compiled LangGraph application
    """
    logger.info("="*80)
    logger.info("INITIALIZING ECU AGENT")
    logger.info("="*80)
    
    # Step 1: Vector store
    logger.info("Step 1/3: Vector Store")
    vectorstore = initialize_vector_store(agent_config, force_rebuild=force_rebuild)
    
    # Step 2: Retriever
    logger.info("Step 2/3: Retriever")
    retriever = initialize_retriever(agent_config)
    
    # Step 3: Graph
    logger.info("Step 3/3: LangGraph Workflow")
    logger.info("Building graph...")
    app = build_graph(retriever, agent_config)
    logger.info("  ✓ Graph ready")
    
    logger.info("="*80)
    logger.info("✓ Agent initialization complete!")
    logger.info("="*80)
    
    return app


# ============================================================
# Evaluation
# ============================================================

def run_evaluation(
    app,
    test_cases: List[Dict[str, str]],
    run_id: str
) -> Dict[str, Any]:
    """
    Run evaluation on test cases.
    
    Args:
        app: LangGraph application
        test_cases: List of test cases
        run_id: MLflow run ID for artifact storage
    
    Returns:
        evaluation_results: Dict with metrics and detailed results
    """
    if not test_cases:
        logger.info("No test cases available, skipping evaluation")
        return {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'pass_rate': 0.0,
            'avg_execution_time': 0.0,
            'total_execution_time': 0.0,
            'results': []
        }
    
    logger.info("="*80)
    logger.info("RUNNING EVALUATION")
    logger.info("="*80)
    logger.info(f"Total test cases: {len(test_cases)}")
    logger.info("")
    
    results = []
    execution_times = []
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case['query']
        logger.info(f"[{i}/{len(test_cases)}] Testing: {query[:60]}...")
        
        start_time = time.time()
        
        try:
            result = app.invoke({"query": query})
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            # Extract result details
            actual_answer = result.get('answer', 'N/A')
            intent_type = result.get('intent', {}).intent_type if hasattr(result.get('intent', {}), 'intent_type') else 'N/A'
            models = str(result.get('intent', {}).models) if hasattr(result.get('intent', {}), 'models') else 'N/A'
            
            results.append({
                'query': query,
                'expected_answer': test_case['expected_answer'],
                'actual_answer': actual_answer,
                'evaluation_criteria': test_case['evaluation_criteria'],
                'intent_type': intent_type,
                'models': models,
                'status': 'PASS',
                'execution_time': execution_time
            })
            
            logger.info(f"  ✓ Completed in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            logger.error(f"  ✗ Failed: {e}")
            
            results.append({
                'query': query,
                'expected_answer': test_case['expected_answer'],
                'actual_answer': f"ERROR: {str(e)}",
                'evaluation_criteria': test_case['evaluation_criteria'],
                'intent_type': 'ERROR',
                'models': 'N/A',
                'status': 'FAIL',
                'execution_time': execution_time
            })
    
    # Calculate metrics
    total_tests = len(test_cases)
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = total_tests - passed
    pass_rate = (passed / total_tests * 100) if total_tests > 0 else 0
    avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
    total_time = sum(execution_times)
    
    logger.info("")
    logger.info("="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"  Passed: {passed} ({pass_rate:.1f}%)")
    logger.info(f"  Failed: {failed} ({(100-pass_rate):.1f}%)")
    logger.info("")
    logger.info("Performance:")
    logger.info(f"  Total time: {format_duration(total_time)}")
    logger.info(f"  Average time: {avg_time:.2f}s per query")
    logger.info("="*80)
    
    # Save detailed results to CSV in artifact directory
    artifact_path = get_artifact_path_for_run(run_id)
    
    # Ensure artifact directory exists
    artifact_path.mkdir(parents=True, exist_ok=True)
    
    results_csv = artifact_path / "evaluation_results.csv"
    
    logger.info(f"Saving detailed results to: {results_csv}")
    
    fieldnames = [
        'query',
        'expected_answer',
        'actual_answer',
        'evaluation_criteria',
        'intent_type',
        'models',
        'status',
        'execution_time'
    ]
    
    with open(results_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info("  ✓ Results saved")
    
    return {
        'total_tests': total_tests,
        'passed': passed,
        'failed': failed,
        'pass_rate': pass_rate,
        'avg_execution_time': avg_time,
        'total_execution_time': total_time,
        'results': results
    }


# ============================================================
# Main Experiment Function
# ============================================================

def run_experiment(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    force_rebuild: bool = False,
    test_csv: Optional[str] = None,
    run_name: Optional[str] = None
) -> str:
    """
    Run complete MLflow experiment.
    
    Args:
        experiment_name: Name of MLflow experiment
        force_rebuild: If True, rebuild vector store
        test_csv: Path to test CSV file
        run_name: Optional name for the MLflow run
    
    Returns:
        run_id: MLflow run ID
    """
    logger.info("="*80)
    logger.info("ECU AGENT - MLFLOW EXPERIMENT")
    logger.info(f"Timestamp: {get_timestamp()}")
    logger.info("="*80)
    logger.info("")
    
    # Setup MLflow experiment
    experiment_id = setup_mlflow_experiment(experiment_name)
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info("")
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run: {run_id}")
        logger.info(f"Run name: {run.info.run_name}")
        logger.info("")
        
        try:
            # Get agent configuration
            agent_config = get_agent_config()
            
            # Log parameters
            logger.info("Logging parameters to MLflow...")
            log_params({
                'embedding_model': agent_config['embedding_model'],
                'chunk_size': agent_config['chunk_size'],
                'chunk_overlap': agent_config['chunk_overlap'],
                'default_top_k': agent_config['default_top_k'],
                'generic_top_k': agent_config['generic_top_k'],
                'compare_top_k': agent_config['compare_top_k'],
                'single_model_top_k': agent_config['single_model_top_k'],
                'force_rebuild': force_rebuild,
            })
            logger.info("  ✓ Parameters logged")
            logger.info("")
            
            # Initialize agent
            start_time = time.time()
            app = initialize_agent(agent_config, force_rebuild=force_rebuild)
            init_time = time.time() - start_time
            
            logger.info(f"Initialization time: {format_duration(init_time)}")
            logger.info("")
            
            # Log initialization metrics
            log_metrics({
                'init_time_seconds': init_time,
            })
            
            # Load test cases
            project_root = Path(__file__).parent.parent.resolve()             
            test_csv_path = project_root / "data" / test_csv
            
            test_cases = load_test_questions(test_csv_path)
            logger.info("")
            
            # Run evaluation
            eval_results = run_evaluation(app, test_cases, run_id)
            logger.info("")
            
            # Log evaluation metrics
            if eval_results['total_tests'] > 0:
                logger.info("Logging evaluation metrics to MLflow...")
                log_metrics({
                    'total_tests': eval_results['total_tests'],
                    'tests_passed': eval_results['passed'],
                    'tests_failed': eval_results['failed'],
                    'pass_rate': eval_results['pass_rate'],
                    'avg_execution_time': eval_results['avg_execution_time'],
                    'total_execution_time': eval_results['total_execution_time'],
                })
                logger.info("  ✓ Evaluation metrics logged")
                logger.info("")
            
            # Get artifact path for this run
            artifact_path = get_artifact_path_for_run(run_id)
            logger.info(f"Artifact directory: {artifact_path}")
            logger.info("")
            
            # Create model wrapper
            logger.info("Creating model wrapper...")
            model_wrapper = ECUAgentWrapper()
            logger.info("  ✓ Model wrapper created")
            logger.info("")
            
            # Log model to MLflow
            model_uri = log_model_to_mlflow(
                model_wrapper=model_wrapper,
                agent_config=agent_config,
                artifact_path=artifact_path,
                model_name="ecu_agent"
            )
            logger.info("")
            
            # Add tags
            mlflow.set_tag("model_type", "rag_agent")
            mlflow.set_tag("framework", "langgraph")
            mlflow.set_tag("vector_store", "faiss")
            
            logger.info("="*80)
            logger.info("✓ EXPERIMENT COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            logger.info(f"Run ID: {run_id}")
            logger.info(f"Model URI: {model_uri}")
            logger.info(f"Artifact Path: {artifact_path}")
            logger.info("="*80)
            
            return run_id
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(e))
            raise


# ============================================================
# Command Line Interface
# ============================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ECU Agent MLflow experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiment.py
  python experiment.py --rebuild
  python experiment.py --experiment-name "my_experiment"
  python experiment.py --run-name "baseline_v1"
  python experiment.py --test-csv "my_test_questions.csv"
        """
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help=f'MLflow experiment name (default: {DEFAULT_EXPERIMENT_NAME})'
    )
    
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Optional name for the MLflow run'
    )
    
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Force rebuild vector store'
    )
    
    parser.add_argument(
        '--test-csv',
        type=str,
        default=DEFAULT_TEST_CSV,
        help=f'Path to test CSV file (default: {DEFAULT_TEST_CSV})'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default=LOG_LEVEL,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help=f'Logging level (default: {LOG_LEVEL})'
    )
    
    return parser.parse_args()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    try:
        # Run experiment
        run_id = run_experiment(
            experiment_name=args.experiment_name,
            force_rebuild=args.rebuild,
            test_csv=args.test_csv,
            run_name=args.run_name
        )
        
        logger.info("")
        logger.info("="*80)
        logger.info("EXPERIMENT COMPLETED!")
        logger.info(f"Run ID: {run_id}")
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  1. View results: from mlruns/1/{run_id}/artifacts/evaluation_results.csv")
        logger.info(f"  2. Load model in your code using MLflow in main.py with run ID: {run_id}")

        
    except Exception as e:
        logger.error(f"EXPERIMENT FAILED: {e}")
        exit(1)