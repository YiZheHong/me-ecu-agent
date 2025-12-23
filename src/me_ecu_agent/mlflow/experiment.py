"""
MLflow Evaluation Runner for ME ECU Engineering Assistant

This module mirrors the existing batch-test style (CSV-driven questions, end-to-end
LangGraph invoke), but logs results, params, and metrics to MLflow instead of
writing a local output CSV.

Place this file under:
    src/me_ecu_agent/mlflow/evaluate_agent.py

Notes:
- Reads test cases from a CSV (same schema as the original batch test).
- Does NOT write results to a local CSV; instead logs a JSON artifact to MLflow.
- Provides two evaluation configs:
    1) "baseline"  -> matches the original batch test config
    2) "variant"   -> adjusted chunking + top-k values (rebuilds vector store)

Expected input CSV columns:
    - Question
    - Expected_Answer
    - Evaluation_Criteria
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow

from me_ecu_agent import query
from me_ecu_agent.ingest import IngestConfig, ingest
from me_ecu_agent.query import QueryFactory
from me_ecu_agent.agent.graph import build_graph
from me_ecu_agent.llm.llm_util import build_eval_prompt, run_llm

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Models
# =============================================================================

@dataclass(frozen=True)
class EvalConfig:
    """Evaluation configuration that controls ingest + retrieval behavior."""
    name: str

    # Embedding & chunking
    embedding_model: str
    chunk_size: int
    chunk_overlap: int

    # Query / retrieval parameters
    default_top_k: int
    default_threshold_score: float
    retrieval_buffer_k: int

    # Agent query settings
    generic_top_k: int
    compare_top_k: int
    single_model_top_k: int
    spec_top_k: int
    max_chars_per_model: int
    max_chars_per_model_specs: int


def baseline_eval_config() -> EvalConfig:
    """Baseline config (matches the original batch test script)."""
    return EvalConfig(
        name="baseline",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=1500,
        chunk_overlap=150,
        default_top_k=3,
        default_threshold_score=1.5,
        retrieval_buffer_k=40,
        generic_top_k=5,
        compare_top_k=3,
        single_model_top_k=3,
        spec_top_k=2,
        max_chars_per_model=1000,
        max_chars_per_model_specs=1000,
    )


def variant_eval_config() -> EvalConfig:
    """
    Variant config (intentionally changes chunking & top-k values).

    Rationale:
    - Smaller chunks may improve pinpoint retrieval for spec tables.
    - Slightly higher top-k can improve recall for comparisons.
    - Adjust buffer to keep performance bounded.
    """
    return EvalConfig(
        name="variant_small_chunks_higher_recall",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=900,
        chunk_overlap=120,
        default_top_k=4,
        default_threshold_score=1.5,
        retrieval_buffer_k=25,
        generic_top_k=6,
        compare_top_k=4,
        single_model_top_k=4,
        spec_top_k=3,
        max_chars_per_model=650,
        max_chars_per_model_specs=1200,
    )


# =============================================================================
# Utilities
# =============================================================================

def setup_logging(level: str = "INFO") -> None:
    """Configure logging with consistent formatting."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def find_project_root(start: Path) -> Path:
    """
    Find the project root by searching for pyproject.toml upwards.
    Falls back to a reasonable parent if not found.

    Args:
        start: File path or directory path to start searching from.

    Returns:
        Path to project root directory.
    """

    return Path(__file__).parents[3].resolve()


def build_agent_config(project_root: Path, eval_cfg: EvalConfig) -> Dict[str, Any]:
    """
    Build a unified agent configuration dict used by ingest + query + graph.

    Args:
        project_root: Project root directory.
        eval_cfg: Evaluation config.

    Returns:
        Dictionary config compatible with IngestConfig.from_dict and QueryFactory.from_dict.
    """
    return {
        # Paths
        "project_root": project_root,
        "data_dir": project_root / "data",
        "vector_dir": project_root / "src" / "me_ecu_agent" / "rag",
        "meta_dir": project_root / "src" / "me_ecu_agent" / "meta",

        # Embedding & chunking
        "embedding_model": eval_cfg.embedding_model,
        "chunk_size": eval_cfg.chunk_size,
        "chunk_overlap": eval_cfg.chunk_overlap,

        # Query parameters
        "default_top_k": eval_cfg.default_top_k,
        "default_threshold_score": eval_cfg.default_threshold_score,
        "retrieval_buffer_k": eval_cfg.retrieval_buffer_k,

        # Agent query settings
        "generic_top_k": eval_cfg.generic_top_k,
        "compare_top_k": eval_cfg.compare_top_k,
        "single_model_top_k": eval_cfg.single_model_top_k,
        "spec_top_k": eval_cfg.spec_top_k,
        "max_chars_per_model": eval_cfg.max_chars_per_model,
        "max_chars_per_model_specs": eval_cfg.max_chars_per_model_specs,
    }


def read_test_questions(csv_path: Path) -> List[Dict[str, str]]:
    """
    Read test questions from CSV (same schema as original batch test).

    Required columns:
        - Question
        - Expected_Answer
        - Evaluation_Criteria

    Args:
        csv_path: Input CSV path.

    Returns:
        List of test case dicts.
    """
    import csv  # local import to keep module load lightweight

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    test_cases: List[Dict[str, str]] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        required = {"Question", "Expected_Answer", "Evaluation_Criteria"}
        missing = required.difference(set(fieldnames))
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            test_cases.append(
                {
                    "question": (row.get("Question") or "").strip(),
                    "expected_answer": (row.get("Expected_Answer") or "").strip(),
                    "evaluation_criteria": (row.get("Evaluation_Criteria") or "").strip(),
                }
            )

    return test_cases


# =============================================================================
# Agent Initialization (mirrors test script behavior)
# =============================================================================

def initialize_agent(agent_config: Dict[str, Any], force_rebuild: bool = False):
    """
    Initialize the full agent pipeline: vector store -> retriever -> graph.

    Args:
        agent_config: Unified config dict.
        force_rebuild: If True, rebuild vector store (recommended if chunk params change).

    Returns:
        Compiled LangGraph application.
    """
    vector_dir = Path(agent_config["vector_dir"])
    faiss_file = vector_dir / "index.faiss"
    pkl_file = vector_dir / "index.pkl"
    exists = faiss_file.exists() and pkl_file.exists()

    if exists and not force_rebuild:
        logger.info("Loading existing vector store...")
        _ = ingest(rebuild=False, config=IngestConfig.from_dict(agent_config))
    else:
        logger.info("Building vector store (rebuild=%s)...", force_rebuild or not exists)
        _ = ingest(rebuild=True, config=IngestConfig.from_dict(agent_config))

    retriever = QueryFactory.from_dict(agent_config)
    app = build_graph(retriever, agent_config)
    return app


# =============================================================================
# Evaluation Logic
# =============================================================================

def _normalize_text(text: str) -> str:
    """Normalize text for simple string-based evaluation."""
    return " ".join(text.lower().split())


def simple_pass_fail(expected_answer: str, actual_answer: str, evaluation_criteria: str) -> bool:
    """
    Minimal heuristic evaluation.

    This is intentionally simple and deterministic. You can later replace this with:
    - MLflow built-in evaluation hooks
    - LLM-as-a-judge
    - domain-specific scoring functions

    Rule:
        PASS if normalized expected answer is a substring of normalized actual answer.

    Args:
        expected_answer: Expected response string.
        actual_answer: Agent response string.

    Returns:
        True if pass else False.
    """
    exp = _normalize_text(expected_answer)
    act = _normalize_text(actual_answer)
    eval_cri = _normalize_text(evaluation_criteria)

    prompt = build_eval_prompt(exp, act, eval_cri)

    answer = run_llm(prompt, expected_answer=exp, actual_answer=act, evaluation_criteria=eval_cri)

    logger.info("Answer from eval LLM: %s", answer)

    return answer.strip().lower() == "pass"


def run_single_test(app, test_case: Dict[str, str]) -> Dict[str, Any]:
    """
    Run a single test case.

    Args:
        app: LangGraph app.
        test_case: Dict with question, expected_answer, evaluation_criteria.

    Returns:
        Result record suitable for artifact logging.
    """
    question = test_case["question"]
    expected = test_case["expected_answer"]
    criteria = test_case["evaluation_criteria"]

    start = time.time()
    try:
        result: Dict[str, Any] = app.invoke({"query": question})
        elapsed = time.time() - start

        actual = str(result.get("answer", ""))
        intent_obj = result.get("intent", None)

        # Safely extract intent fields (matches your original style)
        intent_type = getattr(intent_obj, "intent_type", "N/A")
        models = getattr(intent_obj, "models", "N/A")
        requires_specs = getattr(intent_obj, "requires_specs", "N/A")

        passed = simple_pass_fail(expected, actual, criteria)

        return {
            "question": question,
            "expected_answer": expected,
            "evaluation_criteria": criteria,
            "actual_answer": actual,
            "intent_type": str(intent_type),
            "models": str(models),
            "requires_specs": str(requires_specs),
            "status": "PASS" if passed else "FAIL",
            "execution_time_s": round(elapsed, 3),
        }

    except Exception as exc:
        elapsed = time.time() - start
        return {
            "question": question,
            "expected_answer": expected,
            "evaluation_criteria": criteria,
            "actual_answer": f"ERROR: {exc}",
            "intent_type": "ERROR",
            "models": "N/A",
            "requires_specs": "N/A",
            "status": "FAIL",
            "execution_time_s": round(elapsed, 3),
        }


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate metrics for MLflow logging.

    Args:
        results: Per-test results.

    Returns:
        Summary metrics dict.
    """
    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "PASS")
    failed = total - passed

    times = [float(r.get("execution_time_s", 0.0)) for r in results]
    avg_time = (sum(times) / len(times)) if times else 0.0
    p95_time = _percentile(times, 95) if times else 0.0

    # Intent distribution
    intent_counts: Dict[str, int] = {}
    for r in results:
        intent = str(r.get("intent_type", "N/A"))
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    return {
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": (passed / total) if total else 0.0,
        "avg_latency_s": avg_time,
        "p95_latency_s": p95_time,
        "intent_counts": intent_counts,
    }


def _percentile(values: List[float], pct: int) -> float:
    """Compute percentile without external dependencies."""
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)

    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


# =============================================================================
# MLflow Orchestration
# =============================================================================

def log_results_to_mlflow(
    eval_cfg: EvalConfig,
    agent_config: Dict[str, Any],
    results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    artifact_subdir: str = "eval",
) -> None:
    """
    Log params, metrics, and artifacts to MLflow.

    Args:
        eval_cfg: Eval config object (logged as params).
        agent_config: Unified agent config dict (logged as params with safe keys).
        results: Per-test results (logged as artifact).
        summary: Aggregate metrics (logged as metrics).
        artifact_subdir: Artifact folder name within the run.
    """
    # Params
    mlflow.log_param("eval_config_name", eval_cfg.name)

    # Log EvalConfig fields as params
    for k, v in asdict(eval_cfg).items():
        if k == "name":
            continue
        mlflow.log_param(f"cfg.{k}", v)

    # Log a curated subset of agent_config (paths can be noisy; keep them if you want)
    safe_agent_params = {
        "embedding_model": agent_config.get("embedding_model"),
        "chunk_size": agent_config.get("chunk_size"),
        "chunk_overlap": agent_config.get("chunk_overlap"),
        "default_top_k": agent_config.get("default_top_k"),
        "default_threshold_score": agent_config.get("default_threshold_score"),
        "retrieval_buffer_k": agent_config.get("retrieval_buffer_k"),
        "generic_top_k": agent_config.get("generic_top_k"),
        "compare_top_k": agent_config.get("compare_top_k"),
        "single_model_top_k": agent_config.get("single_model_top_k"),
        "spec_top_k": agent_config.get("spec_top_k"),
        "max_chars_per_model": agent_config.get("max_chars_per_model"),
        "max_chars_per_model_specs": agent_config.get("max_chars_per_model_specs"),
    }
    for k, v in safe_agent_params.items():
        mlflow.log_param(f"agent.{k}", v)

    # Metrics
    mlflow.log_metric("pass_rate", float(summary.get("pass_rate", 0.0)))
    mlflow.log_metric("avg_latency_s", float(summary.get("avg_latency_s", 0.0)))
    mlflow.log_metric("p95_latency_s", float(summary.get("p95_latency_s", 0.0)))
    mlflow.log_metric("total_tests", int(summary.get("total_tests", 0)))
    mlflow.log_metric("passed", int(summary.get("passed", 0)))
    mlflow.log_metric("failed", int(summary.get("failed", 0)))

    # Artifacts: write JSON to a temp file and log it
    artifact_dir = Path("mlruns_artifacts_tmp") / artifact_subdir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = artifact_dir / f"results_{eval_cfg.name}_{timestamp}.json"
    summary_path = artifact_dir / f"summary_{eval_cfg.name}_{timestamp}.json"

    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    mlflow.log_artifact(str(results_path), artifact_path=artifact_subdir)
    mlflow.log_artifact(str(summary_path), artifact_path=artifact_subdir)

    # Intent counts as a JSON param-like artifact (keeps run params clean)
    intent_counts_path = artifact_dir / f"intent_counts_{eval_cfg.name}_{timestamp}.json"
    intent_counts_path.write_text(
        json.dumps(summary.get("intent_counts", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    mlflow.log_artifact(str(intent_counts_path), artifact_path=artifact_subdir)


def run_mlflow_eval(
    experiment_name: str,
    input_csv: str,
    eval_cfg: EvalConfig,
    run_name: Optional[str] = None,
    force_rebuild: bool = False,
    log_level: str = "INFO",
) -> Dict[str, Any]:
    """
    Run a full MLflow evaluation using a given EvalConfig.

    Args:
        experiment_name: MLflow experiment name.
        input_csv: Path to input CSV (relative or absolute).
        eval_cfg: Eval configuration.
        run_name: Optional run name.
        force_rebuild: Force vector store rebuild.
        log_level: Logging level.

    Returns:
        Summary dict including metrics and run_id.
    """
    setup_logging(log_level)
    mlflow.set_experiment(experiment_name)

    # Resolve paths
    this_file = Path(__file__).resolve()
    project_root = find_project_root(this_file)
    csv_path = project_root / "data" / input_csv

    test_cases = read_test_questions(csv_path)
    agent_config = build_agent_config(project_root, eval_cfg)

    # Use MLflow run context
    with mlflow.start_run(run_name=run_name or f"{eval_cfg.name}_{datetime.now().isoformat()}") as run:
        logger.info("MLflow run started: %s", run.info.run_id)
        logger.info("Experiment: %s", experiment_name)
        logger.info("Eval config: %s", eval_cfg.name)
        logger.info("Input CSV: %s", csv_path)

        # Initialize agent (rebuild recommended when chunk params change)
        app = initialize_agent(agent_config, force_rebuild=force_rebuild)

        # Execute tests
        results: List[Dict[str, Any]] = []
        for idx, tc in enumerate(test_cases, start=1):
            logger.info("[%d/%d] %s", idx, len(test_cases), tc["question"][:80])
            results.append(run_single_test(app, tc))

        summary = summarize_results(results)
        log_results_to_mlflow(eval_cfg, agent_config, results, summary)

        logger.info("Done. pass_rate=%.3f avg_latency_s=%.3f p95_latency_s=%.3f",
                    summary["pass_rate"], summary["avg_latency_s"], summary["p95_latency_s"])

        # Return a compact summary
        return {
            "run_id": run.info.run_id,
            "experiment_name": experiment_name,
            "eval_config_name": eval_cfg.name,
            "summary": summary,
        }


def run_two_configs(
    experiment_name: str,
    input_csv: str,
    log_level: str = "INFO",
) -> List[Dict[str, Any]]:
    """
    Run evaluation twice:
      1) baseline config (no rebuild unless needed)
      2) variant config (rebuild recommended due to chunking changes)

    Args:
        experiment_name: MLflow experiment name.
        input_csv: Test CSV path.
        log_level: Logging level.

    Returns:
        List of run summaries.
    """
    base = baseline_eval_config()
    var = variant_eval_config()

    base_summary = run_mlflow_eval(
        experiment_name=experiment_name,
        input_csv=input_csv,
        eval_cfg=base,
        run_name=f"{base.name}",
        force_rebuild=False,
        log_level=log_level,
    )

    var_summary = run_mlflow_eval(
        experiment_name=experiment_name,
        input_csv=input_csv,
        eval_cfg=var,
        run_name=f"{var.name}",
        force_rebuild=True,  # chunking changed -> must rebuild the index
        log_level=log_level,
    )

    return [base_summary, var_summary]


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Example usage:
    #   python -m me_ecu_agent.mlflow.evaluate_agent --exp "ME_ECU_Agent" --csv "test-questions.csv"
    import argparse

    parser = argparse.ArgumentParser(description="Run MLflow evaluation for ME ECU agent.")
    parser.add_argument("--exp", default="basic exp", dest="experiment_name", required=False, help="MLflow experiment name")
    parser.add_argument("--csv", default="test-questions.csv", dest="input_csv", required=False, help="Path to test CSV")
    parser.add_argument("--log-level", dest="log_level", default="INFO", help="Logging level")

    args = parser.parse_args()

    summaries = run_two_configs(
        experiment_name=args.experiment_name,
        input_csv=args.input_csv,
        log_level=args.log_level,
    )

    # Print compact output for CLI usage
    print(json.dumps(summaries, ensure_ascii=False, indent=2))