"""
ECU Agent MLflow Integration

Usage:
    python mlflow_model.py
"""

import mlflow
import mlflow.pyfunc
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from me_ecu_agent.ingest import ingest, IngestConfig
from me_ecu_agent.query import QueryFactory
from me_ecu_agent.agent.graph import build_graph


logger = logging.getLogger(__name__)


# ============================================================
# ECU Agent MLflow Wrapper
# ============================================================

class ECUAgentWrapper(mlflow.pyfunc.PythonModel):
    """MLflow-compatible wrapper for ECU Agent."""
    
    def __init__(self):
        """Initialize empty wrapper (populated during load)."""
        self.app = None
        self.agent_config = None
        self.retriever = None
    
    def load_context(self, context):
        """Load the agent when MLflow loads the model."""
        logger.info("Loading ECU Agent from MLflow artifacts...")
        
        # Load agent config
        config_path = Path(context.artifacts["config"])
        with open(config_path, 'r') as f:
            self.agent_config = json.load(f)
        
        # Reconstruct paths relative to artifact location
        artifact_dir = Path(context.artifacts["vector_store"]).parent
        
        # Update config with artifact paths
        self.agent_config["vector_dir"] = str(artifact_dir / "vector_store")
        self.agent_config["meta_dir"] = str(artifact_dir / "meta_store")
        self.agent_config["meta_db_path"] = str(artifact_dir / "meta_store" / "doc_meta.sqlite")
        
        # Initialize retriever
        logger.info("Initializing retriever...")
        self.retriever = QueryFactory.from_dict(self.agent_config)
        
        # Build graph
        logger.info("Building LangGraph workflow...")
        self.app = build_graph(self.retriever, self.agent_config)
        
        logger.info("✓ ECU Agent loaded successfully")
    
    def predict(self, context, model_input):
        """Run prediction on input queries."""
        # Handle different input types
        if isinstance(model_input, dict):
            queries = [model_input.get("query", "")]
        elif hasattr(model_input, 'to_dict'):
            queries = model_input['query'].tolist()
        else:
            queries = [q.get("query", "") for q in model_input]
        
        logger.info(f"Processing {len(queries)} queries...")
        
        results = []
        for query in queries:
            try:
                result = self.app.invoke({"query": query})
                results.append({
                    "query": query,
                    "answer": result.get('answer', 'N/A'),
                    "intent_type": result.get('intent', {}).intent_type if hasattr(result.get('intent', {}), 'intent_type') else 'N/A',
                    "models": str(result.get('intent', {}).models) if hasattr(result.get('intent', {}), 'models') else 'N/A',
                })
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                results.append({
                    "query": query,
                    "answer": f"ERROR: {str(e)}",
                    "intent_type": "ERROR",
                    "models": "N/A",
                })
        
        # Return simple answers if single query
        if len(results) == 1:
            return results[0]["answer"]
        
        return results


# ============================================================
# Save/Load Functions
# ============================================================

def save_agent_artifacts(app, agent_config: dict, artifact_dir: Path):
    """Save agent artifacts to directory."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving agent artifacts to: {artifact_dir}")
    
    # Save config (with string paths for JSON serialization)
    config_for_save = agent_config.copy()
    for key in ['project_root', 'data_dir', 'vector_dir', 'meta_dir']:
        if key in config_for_save and isinstance(config_for_save[key], Path):
            config_for_save[key] = str(config_for_save[key])
    
    config_path = artifact_dir / "agent_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_for_save, f, indent=2)
    logger.info("  ✓ Config saved")
    
    # Copy vector store files
    vector_src = Path(agent_config["vector_dir"])
    vector_dst = artifact_dir / "vector_store"
    vector_dst.mkdir(exist_ok=True)
    
    shutil.copy2(vector_src / "index.faiss", vector_dst / "index.faiss")
    shutil.copy2(vector_src / "index.pkl", vector_dst / "index.pkl")
    logger.info("  ✓ Vector store saved")
    
    # Copy meta store files (.sqlite)
    meta_src = Path(agent_config["meta_dir"])
    meta_dst = artifact_dir / "meta_store"
    meta_dst.mkdir(exist_ok=True)
    
    for sqlite_file in meta_src.glob("*.sqlite"):
        shutil.copy2(sqlite_file, meta_dst / sqlite_file.name)
    logger.info("  ✓ Meta store saved")
    
    logger.info(f"✓ All artifacts saved to: {artifact_dir}")


def load_agent_from_artifacts(artifact_dir: Path):
    """Load agent from saved artifacts."""
    artifact_dir = Path(artifact_dir)
    logger.info(f"Loading agent from: {artifact_dir}")
    
    # Load config
    config_path = artifact_dir / "agent_config.json"
    with open(config_path, 'r') as f:
        agent_config = json.load(f)
    
    # Update paths to artifact location
    agent_config["vector_dir"] = str(artifact_dir / "vector_store")
    agent_config["meta_dir"] = str(artifact_dir / "meta_store")
    agent_config["meta_db_path"] = str(artifact_dir / "meta_store" / "doc_meta.sqlite")
    
    logger.info("  ✓ Config loaded")
    
    # Initialize retriever
    retriever = QueryFactory.from_dict(agent_config)
    logger.info("  ✓ Retriever initialized")
    
    # Build graph
    app = build_graph(retriever, agent_config)
    logger.info("  ✓ Graph built")
    
    logger.info("✓ Agent loaded successfully")
    return app, agent_config


# ============================================================
# MLflow Integration Functions
# ============================================================

def log_agent_model(
    app,
    agent_config: dict,
    model_name: str = "ecu_agent",
    artifact_dir: Path = None
):
    """Log ECU Agent to MLflow with source code."""
    
    if artifact_dir is None:
        project_root = Path(agent_config.get("project_root", Path.cwd()))
        artifact_dir = project_root / "mlflow_artifacts" / model_name
    
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Using project artifact directory: {artifact_dir}")
    
    # Save artifacts to project directory
    save_agent_artifacts(app, agent_config, artifact_dir)
    
    # Define artifacts for MLflow
    artifacts = {
        "config": str(artifact_dir / "agent_config.json"),
        "vector_store": str(artifact_dir / "vector_store"),
        "meta_store": str(artifact_dir / "meta_store"),
    }
    
    # Get source code path
    project_root = Path(agent_config["project_root"])
    src_path = project_root / "src" / "me_ecu_agent"
    
    mlflow.pyfunc.log_model(
        artifact_path=model_name,
        python_model=ECUAgentWrapper(),
        artifacts=artifacts,
        code_paths=[str(src_path)], 
        pip_requirements=[
            "langchain",
            "langgraph", 
            "faiss-cpu",
            "sentence-transformers",
        ]
    )
    
    run_id = mlflow.active_run().info.run_id
    logger.info(f"✓ Model logged to MLflow (run_id: {run_id})")
    logger.info(f"✓ Source code included from: {src_path}")
    logger.info(f"✓ Artifacts saved in: {artifact_dir}")
    
    return run_id

def load_agent_from_mlflow(run_id: str, model_name: str = "ecu_agent"):
    """Load ECU Agent from MLflow."""
    model_uri = f"runs:/{run_id}/{model_name}"
    logger.info(f"Loading model from MLflow run: {run_id}")
    logger.info(f"Loading model from: {model_uri}")
    
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("✓ Model loaded from MLflow")
    
    return model


# ============================================================
# Example Usage
# ============================================================

def get_agent_config() -> dict:
    """Get unified agent configuration."""
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
    
    return agent_config


def initialize_agent(agent_config: dict, force_rebuild: bool = False):
    """Initialize the complete agent pipeline."""
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


def example_save_and_log():
    """Example: Save agent locally and log to MLflow."""    
    # Initialize agent
    agent_config = get_agent_config()
    app = initialize_agent(agent_config, force_rebuild=False)
    
    # Get project root
    project_root = Path(agent_config["project_root"])
    
    # Save locally
    artifact_dir = project_root / "saved_agent"
    save_agent_artifacts(app, agent_config, artifact_dir)
    print(f"✓ Agent saved to: {artifact_dir}")
    
    # Set MLflow tracking URI to project directory
    tracking_dir = project_root / "mlruns"
    tracking_uri = tracking_dir.as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    print(f"\n✓ MLflow tracking URI: {tracking_uri}")
    print(f"✓ MLflow tracking directory: {tracking_dir}")
    
    # Log to MLflow
    mlflow.set_experiment("ecu_agent_development")
    experiment = mlflow.get_experiment_by_name("ecu_agent_development")
    print(f"✓ Experiment: {experiment.name} (ID: {experiment.experiment_id})")
    
    with mlflow.start_run(run_name="baseline_agent") as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        
        print(f"\n{'='*80}")
        print(f"🔹 RUN CREATED")
        print(f"{'='*80}")
        print(f"Run ID: {run_id}")
        print(f"Experiment ID: {experiment_id}")
        print(f"Run Name: baseline_agent")
        
        # Log config as params
        mlflow.log_param("embedding_model", agent_config["embedding_model"])
        mlflow.log_param("chunk_size", agent_config["chunk_size"])
        mlflow.log_param("chunk_overlap", agent_config["chunk_overlap"])
        
        # Log model
        log_agent_model(app, agent_config, "ecu_agent")
        
        # Show where files are saved
        run_dir = tracking_dir / str(experiment_id) / run_id
        print(f"\n{'='*80}")
        print(f"📁 RUN DIRECTORY LOCATION")
        print(f"{'='*80}")
        print(f"Run directory: {run_dir}")
        print(f"Directory exists: {run_dir.exists()}")
        
        if run_dir.exists():
            print(f"\nContents:")
            for item in sorted(run_dir.iterdir()):
                if item.is_dir():
                    print(f"  📁 {item.name}/")
                else:
                    print(f"  📄 {item.name}")
        
        print(f"\n{'='*80}")
        print(f"✓ Model logged successfully!")
        print(f"{'='*80}")
        print(f"\nTo view in MLflow UI:")
        print(f"  cd {project_root}")
        print(f"  mlflow ui --backend-store-uri {tracking_dir}")
        print(f"  Open: http://localhost:5000")
        print(f"{'='*80}\n")
        
        return run_id


def example_load_and_predict(run_id: str = None):
    """
    Example: Load agent and predict.
    
    Args:
        run_id: MLflow run ID (required if use_local=False)
        use_local: If True, load from saved_agent folder instead of MLflow
    """
    if run_id is None:
        raise ValueError("run_id is required when use_local=False")
    
    model = load_agent_from_mlflow(run_id)
    
    print("✓ Agent loaded successfully from MLflow")
    print()
    
    # Test prediction
    print("Testing prediction...")
    result = model.predict({"query": "What is the CAN bus speed for ECU-850?"})
    print(f"Answer: {result}")

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Save and log agent
    run_id = example_save_and_log()
    
    example_load_and_predict(run_id=run_id)