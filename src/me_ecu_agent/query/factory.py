"""
QueryFactory: Factory pattern for initializing the query system.

This module provides a convenient way to instantiate the complete query
pipeline (Retriever + dependencies) from various configuration sources.

Usage:
    # From QueryConfig
    retriever = QueryFactory.from_config(config)
    
    # From dictionary
    retriever = QueryFactory.from_dict({
        "meta_db_path": "/path/to/doc_meta.sqlite",
        "vectorstore_path": "/path/to/vectorstore",
    })
    
    # From JSON file
    retriever = QueryFactory.from_json_file("config/query.json")
    
    # With default config
    retriever = QueryFactory.create()
"""
from pathlib import Path
from typing import Union, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from me_ecu_agent.query.config import QueryConfig
from me_ecu_agent.query.meta_store import MetaStore
from me_ecu_agent.query.retriever import Retriever


class QueryFactory:
    """
    Factory for creating fully initialized Retriever instances.
    
    This factory handles:
    - Configuration loading from multiple sources
    - VectorStore initialization
    - MetaStore singleton management
    - Complete Retriever instantiation with all dependencies
    
    The factory ensures all components are properly initialized and
    compatible with each other.
    """
    
    @staticmethod
    def _load_vectorstore(
        vectorstore_path: Path,
        embedding_model: str,
    ) -> FAISS:
        """
        Load FAISS vectorstore from disk.
        
        Args:
            vectorstore_path: Path to vectorstore directory
            embedding_model: Embedding model name
        
        Returns:
            FAISS instance
        
        Raises:
            FileNotFoundError: If vectorstore doesn't exist
            ValueError: If path is invalid
        """
        if not vectorstore_path:
            raise ValueError("Vectorstore path cannot be empty")

        if not vectorstore_path.exists():
            raise FileNotFoundError(
                f"Vectorstore not found at: {vectorstore_path}"
            )
        
        if not vectorstore_path.is_dir():
            raise ValueError(
                f"Vectorstore path must be a directory: {vectorstore_path}"
            )
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Load vectorstore
        vectorstore = FAISS.load_local(
            str(vectorstore_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        
        return vectorstore
    
    @staticmethod
    def _initialize_meta_store(config: QueryConfig) -> MetaStore:
        """
        Initialize MetaStore with configuration.
        
        Args:
            config: QueryConfig instance
        
        Returns:
            MetaStore singleton instance
        
        Raises:
            FileNotFoundError: If database doesn't exist
        """
        if not config.meta_dir:
            raise ValueError("Meta directory path cannot be empty")
        
        if not config.meta_dir.exists():
            raise FileNotFoundError(
                f"DocMeta database not found at: {config.meta_dir}"
            )
        
        # MetaStore is a singleton - this will reuse existing instance if present
        meta_store = MetaStore(config)
        
        if not meta_store.is_loaded():
            raise RuntimeError("MetaStore failed to load metadata from database")
        
        return meta_store
    
    @staticmethod
    def create(config: Optional[QueryConfig] = None) -> Retriever:
        """
        Create a Retriever with default or provided configuration.
        
        Args:
            config: QueryConfig instance. If None, uses default config.
        
        Returns:
            Fully initialized Retriever instance
        
        Raises:
            FileNotFoundError: If required files don't exist
            ValueError: If configuration is invalid
        
        Example:
            >>> # Using default config
            >>> retriever = QueryFactory.create()
            
            >>> # Using custom config
            >>> config = QueryConfig(
            ...     meta_db_path=Path("/data/doc_meta.sqlite"),
            ...     vectorstore_path=Path("/data/vectorstore"),
            ... )
            >>> retriever = QueryFactory.create(config)
        """
        if config is None:
            raise ValueError("Configuration cannot be None")
        
        # Validate configuration
        try:
            config.validate()
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Configuration validation failed: {e}")
        
        # Initialize components
        meta_store = QueryFactory._initialize_meta_store(config)
        vectorstore = QueryFactory._load_vectorstore(
            config.vector_dir,
            config.embedding_model_name,
        )
        
        # Create and return Retriever
        retriever = Retriever(vectorstore, meta_store, config)
        return retriever
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> Retriever:
        """
        Create a Retriever from a configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            Fully initialized Retriever instance
        
        Raises:
            ValueError: If configuration is invalid
        
        Example:
            >>> config_dict = {
            ...     "meta_db_path": "/data/doc_meta.sqlite",
            ...     "vectorstore_path": "/data/vectorstore",
            ...     "default_top_k": 5,
            ... }
            >>> retriever = QueryFactory.from_dict(config_dict)
        """
        config = QueryConfig.from_dict(config_dict)
        print(config)
        return QueryFactory.create(config)
        """
        Create a Retriever from a JSON configuration file.
        
        Args:
            json_path: Path to JSON configuration file
        
        Returns:
            Fully initialized Retriever instance
        
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            ValueError: If configuration is invalid
        
        Example:
            >>> retriever = QueryFactory.from_json_file("config/query.json")
        """
        config = QueryConfig.from_json_file(json_path)
        return QueryFactory.create(config)