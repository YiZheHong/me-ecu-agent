"""
main.py

Minimal entry point to load an ECU agent from MLflow and run a prediction.
"""

import sys
from pathlib import Path

# CRITICAL: Add mlflow_agent directory to Python path
# This ensures MLflow can find mlflow_model module when loading
project_root = Path(__file__).parent.resolve()
mlflow_agent_dir = project_root / "mlflow_agent"
if str(mlflow_agent_dir) not in sys.path:
    sys.path.insert(0, str(mlflow_agent_dir))

import mlflow
import mlflow.pyfunc


def load_agent_from_mlflow(run_id: str):
    """Load ECU Agent model from MLflow run."""
    # Configure MLflow
    mlruns_dir = project_root / "mlruns"
    tracking_uri = f"sqlite:///{mlruns_dir / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Load model
    model_uri = f"runs:/{run_id}/ecu_agent"
    print(f"Loading model from: {model_uri}")
    
    model = mlflow.pyfunc.load_model(model_uri)
    print("✓ Model loaded successfully\n")
    
    return model


def main():
    """Main execution function."""
    # Your run ID
    run_id = "8e3d167738654426b990548ddc385dd3"
    
    # Load model
    model = load_agent_from_mlflow(run_id)
    
    # Test prediction
    print("Testing prediction...")
    input_data = {"query": "What is the CAN bus speed for ECU-850?"}
    
    result = model.predict(input_data)
    
    print(f"\nQuery: {input_data['query']}")
    print(f"Answer: {result['answer']}")
    print(f"Intent: {result.get('intent_type', 'N/A')}")
    print(f"Models: {result.get('models', 'N/A')}")


if __name__ == "__main__":
    main()