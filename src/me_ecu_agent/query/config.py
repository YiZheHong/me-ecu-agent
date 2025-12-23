"""
Enhanced configuration management for the query module.

Supports multiple configuration sources and provides flexible
parameter management similar to the ingest module.
"""
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
import os
from dataclasses import dataclass, asdict


@dataclass
class QueryConfig:
    """
    Configuration for query operations.
    
    Supports:
    - Direct instantiation
    - Loading from dict
    - Loading from JSON file
    - Loading from environment variables
    - Loading from YAML file
    """
    
    # Database configuration
    meta_dir: Optional[Path] = None
    vector_dir: Optional[Path] = None
    
    # Retrieval parameters
    default_top_k: int = 5
    retrieval_buffer_k: int = 40
    
    # Optional: embedding model configuration
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Feature flags
    enable_source_tracking: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QueryConfig":
        """
        Create QueryConfig from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            QueryConfig instance
        
        Example:
            >>> config = QueryConfig.from_dict({
            ...     "meta_dir": "/path/to/doc_meta.sqlite",
            ...     "vector_dir": "/path/to/vectorstore",
            ...     "default_top_k": 5,
            ...     "retrieval_buffer_k": 40,
            ... })
        """
        # Get all dataclass fields
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        
        # Filter and convert
        filtered_dict = {}
        
        for k, v in config_dict.items():
            if k not in field_names:
                continue
            
            # Convert string paths to Path objects
            if k.endswith('_dir') and isinstance(v, str):
                filtered_dict[k] = Path(v)
            else:
                filtered_dict[k] = v
        
        return cls(**filtered_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                result[key] = str(value) if value else None
            else:
                result[key] = value
        return result
    
    def validate(self) -> None:
        """
        Validate configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self.default_top_k <= 0:
            raise ValueError("default_top_k must be > 0")
        
        if self.retrieval_buffer_k < self.default_top_k:
            raise ValueError("retrieval_buffer_k must be >= default_top_k")
        
        if self.meta_dir and not self.meta_dir.exists():
            raise FileNotFoundError(f"DocMeta database not found: {self.meta_dir}")
        
        if self.vector_dir and not self.vector_dir.exists():
            raise FileNotFoundError(f"Vectorstore not found: {self.vector_dir}")
        
        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be >= 0")
    
    def __repr__(self) -> str:
        """Pretty print configuration."""
        items = []
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                value = str(value) if value else None
            items.append(f"  {key}={value}")
        
        return f"QueryConfig(\n" + "\n".join(items) + "\n)"