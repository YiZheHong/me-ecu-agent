"""
Test script for query_chunks_by_model.

This script demonstrates usage and validates the retrieval functionality.
"""
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from me_ecu_agent.query import (
    QueryConfig,
    query_chunks_by_model,
    query_chunks_generic,
    get_meta_store,
)


def setup_test_environment():
    """
    Setup the test environment.
    
    This includes:
    - Loading the vector store
    - Configuring paths
    - Initializing the MetaStore
    """
    project_root = Path(__file__).parents[0].resolve()
    print(f"Project root detected at: {project_root}")
    
    # Configure paths
    config = QueryConfig(
        meta_db_path=project_root / "src" / "me_ecu_agent" / "meta" / "doc_meta.sqlite",
        default_top_k=5,
        retrieval_buffer_k=40,
    )
    
    # Validate configuration
    config.validate()
    
    # Load vector store
    vectorstore_path = project_root / "src" / "me_ecu_agent" / "rag"
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    
    # Initialize MetaStore
    meta_store = get_meta_store(config)
    
    print(f"✓ Loaded {len(meta_store.get_all())} document metadata entries")
    print(f"✓ Vector store loaded from {vectorstore_path}")
    
    return vectorstore, config


def test_single_model_query(vectorstore, config):
    """
    Test query_chunks_by_model with various models.
    """
    print("\n" + "="*60)
    print("TEST: Single Model Queries")
    print("="*60)
    
    test_cases = [
        ("What is the maximum operating temperature for the ECU-750?", "ECU-750"),
        ("How much RAM does the ECU-850 have?", "ECU-850"),
        ("What are the AI capabilities of the ECU-850b?", "ECU-850b"),
    ]
    
    for query, model in test_cases:
        print(f"\nQuery: {query}")
        print(f"Model: {model}")
        print("-" * 60)
        
        results = query_chunks_by_model(
            query=query,
            model=model,
            vectorstore=vectorstore,
            config=config,
        )
        
        print(f"Retrieved {len(results)} chunks:")
        for i, (doc, score) in enumerate(results[:3], 1):
            print(f"\n[{i}] Score: {score:.4f}")
            print(f"Doc UID: {doc.metadata.get('doc_uid', 'N/A')}")
            print(f"Content: {doc.page_content[:200]}...")

def test_nonexistent_model(vectorstore, config):
    """
    Test behavior when querying a non-existent model.
    """
    print("\n" + "="*60)
    print("TEST: Non-existent Model")
    print("="*60)
    
    query = "What is the temperature range?"
    model = "ECU-999"  # Doesn't exist
    
    print(f"\nQuery: {query}")
    print(f"Model: {model} (non-existent)")
    print("-" * 60)
    
    results = query_chunks_by_model(
        query=query,
        model=model,
        vectorstore=vectorstore,
        config=config,
    )
    
    print(f"Retrieved {len(results)} chunks")
    print("Expected: 0 (no documents cover this model)")


def test_generic_query(vectorstore, config):
    """
    Test query_chunks_generic for general technical questions.
    """
    print("\n" + "="*60)
    print("TEST: Generic Query (No Model Constraint)")
    print("="*60)
    
    query = "What is CAN bus used for?"
    
    print(f"\nQuery: {query}")
    print("-" * 60)
    
    results = query_chunks_generic(
        query=query,
        vectorstore=vectorstore,
        config=config,
        top_k=5,
    )
    
    print(f"Retrieved {len(results)} chunks:")
    for i, (doc, score) in enumerate(results[:3], 1):
        print(f"\n[{i}] Score: {score:.4f}")
        print(f"Doc UID: {doc.metadata.get('doc_uid', 'N/A')}")
        print(f"Content: {doc.page_content[:200]}...")


def main():
    """
    Run all tests.
    """
    print("Setting up test environment...")
    vectorstore, config = setup_test_environment()
    
    # Run test suites
    # test_single_model_query(vectorstore, config)
    # test_nonexistent_model(vectorstore, config)
    test_generic_query(vectorstore, config)
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()