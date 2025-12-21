"""
Test script for the ECU agent.

Tests all three query types:
1. Single model queries
2. Comparison queries
3. Generic queries
"""
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from me_ecu_agent.agent import build_graph
from me_ecu_agent.query import QueryConfig, set_default_config


def setup_test_environment():
    """
    Setup vectorstore and configuration.
    """
    project_root = Path(__file__).parents[0].resolve()
    print(f"Project root detected at: {project_root}")
    
    config = QueryConfig(
        meta_db_path=project_root / "src" / "me_ecu_agent" / "meta" / "doc_meta.sqlite",
        default_top_k=5,
        retrieval_buffer_k=40,
    )
    
    set_default_config(config)
    
    # Load vectorstore
    vectorstore_path = project_root / "src" / "me_ecu_agent" / "rag"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    
    print(f"✓ Vector store loaded from {vectorstore_path}")
    
    return vectorstore


def test_single_model_queries(app):
    """
    Test single model queries.
    """
    print("\n" + "="*60)
    print("TEST: Single Model Queries")
    print("="*60)
    
    test_cases = [
        "What is the maximum operating temperature for the ECU-750?",
        "How much RAM does the ECU-850 have?",
        "What are the AI capabilities of the ECU-850b?",
    ]
    
    for query in test_cases:
        print(f"\n📝 Query: {query}")
        print("-" * 60)
        
        result = app.invoke({"query": query})
        
        print(f"\n💡 Answer:\n{result['answer']}\n")


def test_comparison_queries(app):
    """
    Test comparison queries (concurrent retrieval).
    """
    print("\n" + "="*60)
    print("TEST: Comparison Queries")
    print("="*60)
    
    test_cases = [
        "What are the differences between ECU-850 and ECU-850b?",
        "Compare the CAN bus capabilities of ECU-750 and ECU-850.",
        "Which has better AI capabilities, ECU-850 or ECU-850b?",
    ]
    
    for query in test_cases:
        print(f"\n📝 Query: {query}")
        print("-" * 60)
        
        result = app.invoke({"query": query})
        
        print(f"\n💡 Answer:\n{result['answer']}\n")


def test_generic_queries(app):
    """
    Test generic queries (no model specified).
    """
    print("\n" + "="*60)
    print("TEST: Generic Queries")
    print("="*60)
    
    test_cases = [
        "What is CAN bus used for?",
        "How does Over-the-Air update work?",
        "What is an NPU?",
    ]
    
    for query in test_cases:
        print(f"\n📝 Query: {query}")
        print("-" * 60)
        
        result = app.invoke({"query": query})
        
        print(f"\n💡 Answer:\n{result['answer']}\n")


def test_edge_cases(app):
    """
    Test edge cases and special scenarios.
    """
    print("\n" + "="*60)
    print("TEST: Edge Cases")
    print("="*60)
    
    test_cases = [
        # Multiple models mentioned but no comparison intent
        "The ECU-750 and ECU-850 both use CAN bus. What is it?",
        
        # Non-existent model
        "What is the temperature range for ECU-999?",
        
        # Cross-model filtering query
        "Which ECU models support OTA updates?",
    ]
    
    for query in test_cases:
        print(f"\n📝 Query: {query}")
        print("-" * 60)
        
        result = app.invoke({"query": query})
        
        print(f"\n💡 Answer:\n{result['answer']}\n")

def visualize_graph(app, output_path="agent_graph.png"):
    """
    Visualize the LangGraph workflow and save to file.
    
    Args:
        app: Compiled LangGraph application
        output_path: Path to save the graph image
    """
    try:
        from IPython.display import Image, display
        
        # Generate graph image
        graph_image = app.get_graph().draw_mermaid_png()
        
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(graph_image)
        
        print(f"✓ Graph visualization saved to: {output_path}")
        
        # Try to display if in Jupyter
        try:
            display(Image(graph_image))
        except:
            print("  (Not in Jupyter environment, image saved to file only)")
            
    except ImportError:
        print("⚠ Visualization requires: pip install pygraphviz")
    except Exception as e:
        print(f"⚠ Could not generate graph visualization: {e}")
        print("  Try installing: pip install pygraphviz")

def main():
    """
    Run all tests.
    """
    print("Setting up test environment...")
    vectorstore = setup_test_environment()
    
    print("Building agent graph...")
    app = build_graph(vectorstore)
    print("✓ Agent ready\n")

    # Visualize the graph
    print("Generating graph visualization...")
    visualize_graph(app, output_path="./agent_graph.png")
    print()
    
    # # Run test suites
    # test_single_model_queries(app)
    # test_comparison_queries(app)
    # test_generic_queries(app)
    # test_edge_cases(app)
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()