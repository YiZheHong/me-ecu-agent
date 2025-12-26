from setuptools import setup, find_packages

setup(
    name="me-ecu-agent",
    version="0.1.0",
    description="ECU Electronic Control Unit document retrieval agent",
    author="Potter Hong",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langchain-community>=0.0.1",
        "langgraph>=0.0.1",
        "langchain-huggingface>=1.0.0",  # HuggingFace embeddings
        "langchain-openai>=1.0.0",       # OpenAI LLM

        # Embeddings
        "sentence-transformers>=2.0.0",
        
        # Vector Store
        "faiss-cpu>=1.7.4",
        
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "ipython>=9.0.0",
            "pytest>=7.0",
            "black",
            "isort",
        ]
    },

    package_data={
        "me_ecu_agent.meta": ["*.sqlite"],
    },
    include_package_data=True,
)