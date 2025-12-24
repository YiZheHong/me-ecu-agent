"""
MLflow Utility Functions - COMPLETE FIX

Configures MLflow with:
- FileStore backend for run metadata (meta.yaml, params/, etc.)
- SQLite backend for Model Registry
- FileSystem for artifacts

This gives you BOTH:
1. Traditional mlruns/ directory structure with meta.yaml files
2. Model Registry functionality
"""

import logging
import mlflow
import mlflow.pyfunc
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from me_ecu_agent.ingest import ingest, IngestConfig
from me_ecu_agent.query import QueryFactory


logger = logging.getLogger(__name__)


# ============================================================
# CRITICAL: Configure MLflow Backend
# ============================================================

def configure_mlflow_backend():
    """
    Configure MLflow with FileStore + SQLite hybrid backend.
    
    This configuration provides:
    1. FileStore for run metadata (meta.yaml, params/, metrics/, tags/)
    2. SQLite for Model Registry (models/)
    3. FileSystem for artifacts
    
    Call this BEFORE any other MLflow operations!
    """
    # Get project root
    project_root = Path(__file__).parent.parent.resolve()
    mlruns_dir = project_root / "mlruns"
    
    # Create mlruns directory
    mlruns_dir.mkdir(exist_ok=True)
    
    # CRITICAL: Use sqlite:/// URI with mlruns path
    # This enables BOTH FileStore AND Model Registry
    tracking_uri = f"sqlite:///{mlruns_dir / 'mlflow.db'}"
    
    mlflow.set_tracking_uri(tracking_uri)
    
    logger.info("="*80)
    logger.info("MLflow Backend Configuration")
    logger.info("="*80)
    logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"Backend Store: SQLite (for metadata + model registry)")
    logger.info(f"Artifact Store: FileSystem ({mlruns_dir})")
    logger.info(f"Database: {mlruns_dir / 'mlflow.db'}")
    logger.info("="*80)
    
    return mlruns_dir


# ============================================================
# Configuration
# ============================================================

def get_agent_config() -> Dict[str, Any]:
    """
    Get unified agent configuration for both ingest and query.
    
    Returns:
        agent_config: Single config dict for all operations
    """
    # Get project root - go up from mlflow/ to project root
    # mlflow/mlflow_utils.py -> mlflow/ -> project_root/
    project_root = Path(__file__).parent.parent.resolve()
    
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
# Vector Store Initialization
# ============================================================

def initialize_vector_store(agent_config: Dict[str, Any], force_rebuild: bool = False):
    """
    Initialize vector store (build or load existing).
    
    Args:
        agent_config: Agent configuration dictionary
        force_rebuild: If True, rebuild vector store from scratch
    
    Returns:
        vectorstore: FAISS vectorstore instance
    """
    logger.info("Initializing vector store...")
    
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
    return vectorstore


def initialize_retriever(agent_config: Dict[str, Any]):
    """
    Initialize retriever from agent config.
    
    Args:
        agent_config: Agent configuration dictionary
    
    Returns:
        retriever: QueryFactory retriever instance
    """
    logger.info("Initializing retriever...")
    retriever = QueryFactory.from_dict(agent_config)
    logger.info("  ✓ Retriever ready")
    return retriever


# ============================================================
# Artifact Management
# ============================================================

def prepare_artifacts_for_mlflow(
    agent_config: Dict[str, Any],
    artifact_path: Path
) -> Dict[str, str]:
    """
    Copy necessary artifacts to MLflow artifact directory.
    
    Args:
        agent_config: Agent configuration dictionary
        artifact_path: Path to MLflow artifact directory
    
    Returns:
        artifacts: Dict mapping artifact names to paths
    """
    logger.info(f"Preparing artifacts in: {artifact_path}")
    logger.debug(f"Artifact path type: {type(artifact_path)}")
    logger.debug(f"Artifact path exists: {artifact_path.exists() if hasattr(artifact_path, 'exists') else 'N/A'}")
    
    # Ensure artifact_path is a proper Path object
    artifact_path = Path(artifact_path)
    
    # Create main artifact directory
    try:
        artifact_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created artifact directory: {artifact_path}")
    except Exception as e:
        logger.error(f"Failed to create artifact directory: {artifact_path}")
        logger.error(f"Error: {e}")
        raise
    
    # Create subdirectories
    vector_store_dst = artifact_path / "vector_store"
    meta_store_dst = artifact_path / "meta_store"
    
    vector_store_dst.mkdir(exist_ok=True)
    meta_store_dst.mkdir(exist_ok=True)
    
    # Copy vector store files
    vector_dir = Path(agent_config["vector_dir"])
    logger.info("  Copying vector store...")
    
    for file in ["index.faiss", "index.pkl"]:
        src = vector_dir / file
        dst = vector_store_dst / file
        if src.exists():
            shutil.copy2(src, dst)
            logger.debug(f"    Copied {file}")
        else:
            raise FileNotFoundError(f"Vector store file not found: {src}")
    
    # Copy metadata database
    meta_dir = Path(agent_config["meta_dir"])
    meta_db = meta_dir / "doc_meta.sqlite"
    logger.info("  Copying metadata store...")
    
    if meta_db.exists():
        shutil.copy2(meta_db, meta_store_dst / "doc_meta.sqlite")
        logger.debug(f"    Copied doc_meta.sqlite")
    else:
        raise FileNotFoundError(f"Metadata database not found: {meta_db}")
    
    # Save agent config as JSON
    config_file = artifact_path / "agent_config.json"
    logger.info("  Saving agent config...")
    
    # Convert Path objects to strings for JSON serialization
    config_for_json = {}
    for key, value in agent_config.items():
        if isinstance(value, Path):
            config_for_json[key] = str(value)
        else:
            config_for_json[key] = value
    
    with open(config_file, 'w') as f:
        json.dump(config_for_json, f, indent=2)
    logger.debug(f"    Saved agent_config.json")
    
    artifacts = {
        "vector_store": str(vector_store_dst),
        "meta_store": str(meta_store_dst),
        "config": str(config_file),
    }
    
    logger.info("  ✓ Artifacts prepared")
    return artifacts


# ============================================================
# MLflow Model Logging
# ============================================================

def log_model_to_mlflow(
    model_wrapper,
    agent_config: Dict[str, Any],
    artifact_path: Path,
    model_name: str = "ecu_agent"
) -> str:
    """
    Log ECU Agent model to MLflow.
    
    Args:
        model_wrapper: ECUAgentWrapper instance
        agent_config: Agent configuration dictionary
        artifact_path: Path to artifact directory (run's artifacts directory)
        model_name: Name for the model
    
    Returns:
        model_uri: MLflow model URI
    """
    logger.info("Logging model to MLflow...")
    
    # Create temporary directory for artifacts
    temp_dir = Path(tempfile.mkdtemp())
    logger.debug(f"Created temp directory: {temp_dir}")
    
    try:
        # Create subdirectories in temp
        vector_store_temp = temp_dir / "vector_store"
        meta_store_temp = temp_dir / "meta_store"
        vector_store_temp.mkdir()
        meta_store_temp.mkdir()
        
        # Copy ONLY the data files (not Python code)
        vector_dir = Path(agent_config["vector_dir"])
        meta_dir = Path(agent_config["meta_dir"])
        
        # Copy vector store files
        logger.info("  Preparing vector store...")
        for file in ["index.faiss", "index.pkl"]:
            src = vector_dir / file
            dst = vector_store_temp / file
            if src.exists():
                shutil.copy2(src, dst)
                logger.debug(f"    Copied {file}")
            else:
                raise FileNotFoundError(f"Vector store file not found: {src}")
        
        # Copy metadata database
        logger.info("  Preparing metadata store...")
        meta_db = meta_dir / "doc_meta.sqlite"
        if meta_db.exists():
            shutil.copy2(meta_db, meta_store_temp / "doc_meta.sqlite")
            logger.debug(f"    Copied doc_meta.sqlite")
        else:
            raise FileNotFoundError(f"Metadata database not found: {meta_db}")
        
        # Save agent config
        logger.info("  Preparing agent config...")
        config_file = temp_dir / "agent_config.json"
        config_for_json = {}
        for key, value in agent_config.items():
            if isinstance(value, Path):
                config_for_json[key] = str(value)
            else:
                config_for_json[key] = value
        
        with open(config_file, 'w') as f:
            json.dump(config_for_json, f, indent=2)
        logger.debug(f"    Saved agent_config.json")
        
        # Define artifacts dictionary - point to temp directory
        artifacts = {
            "vector_store": str(vector_store_temp),
            "meta_store": str(meta_store_temp),
            "config": str(config_file),
        }
        
        logger.info(f"  Artifacts prepared in: {temp_dir}")
        
        # Define pip requirements
        pip_requirements = [
            "mlflow",
            "langchain",
            "langchain-openai", 
            "langgraph",
            "faiss-cpu",
            "sentence-transformers",
            "pandas",
        ]
        
        # Log model - use 'registered_model_name' parameter to control where it goes
        # If registered_model_name is None, it saves to run artifacts only
        logger.info(f"  Logging model to MLflow...")
        model_info = mlflow.pyfunc.log_model(
            artifact_path=model_name,          # Where in run artifacts to save
            python_model=model_wrapper,
            artifacts=artifacts,
            pip_requirements=pip_requirements,
            registered_model_name=None,        # Don't auto-register to Model Registry
        )
        
        logger.info(f"  ✓ Model logged: {model_info.model_uri}")
        return model_info.model_uri
        
    finally:
        # Clean up temp directory
        logger.debug(f"Cleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir)


# ============================================================
# Metrics Logging
# ============================================================

def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """
    Log metrics to MLflow.
    
    Args:
        metrics: Dictionary of metric name -> value
        step: Optional step number for the metrics
    """
    for name, value in metrics.items():
        mlflow.log_metric(name, value, step=step)
    
    logger.debug(f"Logged metrics: {list(metrics.keys())}")


def log_params(params: Dict[str, Any]):
    """
    Log parameters to MLflow.
    
    Args:
        params: Dictionary of parameter name -> value
    """
    for name, value in params.items():
        # Convert Path to string
        if isinstance(value, Path):
            value = str(value)
        mlflow.log_param(name, value)
    
    logger.debug(f"Logged params: {list(params.keys())}")


# ============================================================
# Experiment Management
# ============================================================

def setup_mlflow_experiment(
    experiment_name: str,
    tracking_uri: Optional[str] = None
) -> str:
    """
    Setup MLflow experiment.
    
    Args:
        experiment_name: Name of the experiment
        tracking_uri: Optional tracking URI (defaults to SQLite + FileStore)
    
    Returns:
        experiment_id: ID of the experiment
    """
    # CRITICAL: Configure backend if tracking_uri not specified
    if tracking_uri is None:
        configure_mlflow_backend()
    else:
        mlflow.set_tracking_uri(tracking_uri)
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
    
    mlflow.set_experiment(experiment_name)
    return experiment_id


def get_artifact_path_for_run(run_id: str) -> Path:
    """
    Get the artifact path for a specific MLflow run.
    
    Args:
        run_id: MLflow run ID
    
    Returns:
        artifact_path: Path to the run's artifact directory
    """
    import platform
    from urllib.parse import urlparse, unquote
    
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    artifact_uri = run.info.artifact_uri
    
    # Parse URI properly for both Windows and Unix
    if artifact_uri.startswith("file://"):
        parsed = urlparse(artifact_uri)
        path_str = unquote(parsed.path)
        
        # Windows fix: Remove leading slash from /C:/...
        if platform.system() == "Windows" and path_str.startswith("/") and ":" in path_str:
            path_str = path_str[1:]  # Remove leading /
        
        artifact_path = Path(path_str)
    else:
        artifact_path = Path(artifact_uri)
    
    return artifact_path


# ============================================================
# Timestamp Utilities
# ============================================================

def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"