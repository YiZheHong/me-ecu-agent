from me_ecu_agent.mlflow.mlflow_model import load_agent_from_mlflow

model = load_agent_from_mlflow("c273982e66ee4dcf93faf67ff9ae8026")
    
print("✓ Agent loaded successfully from MLflow")
print()

# Test prediction
print("Testing prediction...")
result = model.predict({"query": "What is the CAN bus speed for ECU-850?"})
print(f"Answer: {result}")