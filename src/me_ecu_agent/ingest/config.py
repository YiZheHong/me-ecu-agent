"""
Configuration module for ME ECU Agent ingestion
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class IngestConfig:
    """
    Configuration for document ingestion pipeline.
    
    This class manages all parameters needed for:
    - Embedding model selection
    - Chunking strategy
    - File paths
    - Database settings
    """
    
    # Embedding configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Chunking configuration
    chunk_size: int = 1500
    chunk_overlap: int = 300
    
    # Path configuration
    project_root: Optional[Path] = None
    data_dir: Optional[Path] = None
    vector_dir: Optional[Path] = None
    meta_dir: Optional[Path] = None
    
    # Database configuration
    meta_db_name: str = "doc_meta.sqlite"
    
    @property
    def meta_db_path(self) -> Path:
        """Get full path to metadata database."""
        return self.meta_dir / self.meta_db_name
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "IngestConfig":
        """
        Create IngestConfig from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            IngestConfig instance
            
        Example:
            >>> config = IngestConfig.from_dict({
            ...     "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            ...     "chunk_size": 500,
            ...     "chunk_overlap": 50,
            ... })
        """
        # Filter out keys that aren't IngestConfig parameters
        valid_keys = {
            'embedding_model', 'chunk_size', 'chunk_overlap',
            'project_root', 'data_dir', 'vector_dir', 'meta_dir',
            'meta_db_name'
        }
        
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        # Convert string paths to Path objects
        for path_key in ['project_root', 'data_dir', 'vector_dir', 'meta_dir']:
            if path_key in filtered_dict and isinstance(filtered_dict[path_key], str):
                filtered_dict[path_key] = Path(filtered_dict[path_key])
        
        return cls(**filtered_dict)
    
    def to_dict(self) -> dict:
        """
        Convert config to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        return {
            'embedding_model': self.embedding_model,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'project_root': str(self.project_root),
            'data_dir': str(self.data_dir),
            'vector_dir': str(self.vector_dir),
            'meta_dir': str(self.meta_dir),
            'meta_db_name': self.meta_db_name,
        }
    
    def __repr__(self) -> str:
        """Pretty print configuration."""
        return (
            f"IngestConfig(\n"
            f"  embedding_model='{self.embedding_model}',\n"
            f"  chunk_size={self.chunk_size},\n"
            f"  chunk_overlap={self.chunk_overlap},\n"
            f"  project_root={self.project_root},\n"
            f"  data_dir={self.data_dir},\n"
            f"  vector_dir={self.vector_dir},\n"
            f"  meta_db_path={self.meta_db_path}\n"
            f")"
        )