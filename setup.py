from setuptools import setup, find_packages

setup(
    name="me-ecu-agent",
    version="0.1.0",
    description="ECU Electronic Control Unit document retrieval agent",
    author="Potter Hong",
    python_requires=">=3.10",
    package_dir={"": "src"},  # Tell setuptools to look in src/ directory
    packages=find_packages(where="src"),  # Find packages in src/
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.1",
        "langchain-core>=0.1.0",
        "faiss-cpu>=1.7.4",
        "langgraph>=0.0.1",
    ],
)