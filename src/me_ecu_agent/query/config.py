"""
Configuration management for the query module.

Centralizes path configuration and retrieval parameters.
"""
from pathlib import Path
from typing import Optional


class QueryConfig:
    """
    Configuration for query operations.
    
    This class manages paths and retrieval parameters in a centralized way,
    avoiding hardcoded paths and magic numbers scattered across the codebase.
    """
    
    def __init__(
        self,
        meta_db_path: Optional[Path] = None,
        default_top_k: int = 5,
        retrieval_buffer_k: int = 40,
    ):
        """
        Args:
            meta_db_path: Path to doc_meta.sqlite. If None, auto-detect.
            default_top_k: Default number of chunks to return.
            retrieval_buffer_k: Initial retrieval size before filtering.
        """
        self.meta_db_path = meta_db_path or self._auto_detect_meta_db_path()
        self.default_top_k = default_top_k
        self.retrieval_buffer_k = retrieval_buffer_k
    
    @staticmethod
    def _auto_detect_meta_db_path() -> Path:
        """
        Auto-detect the path to doc_meta.sqlite.
        
        Tries multiple common locations:
        1. Project root structure (for development)
        2. Current working directory (for deployment)
        
        Returns:
            Path to doc_meta.sqlite
            
        Raises:
            FileNotFoundError: If database cannot be located.
        """
        # Try project structure: src/me_ecu_agent/meta/doc_meta.sqlite
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3]  # me_ecu_agent/query/config.py -> root
        
        candidate_paths = [
            project_root / "src" / "me_ecu_agent" / "meta" / "doc_meta.sqlite",
            project_root / "me_ecu_agent" / "meta" / "doc_meta.sqlite",
            Path.cwd() / "meta" / "doc_meta.sqlite",
        ]
        
        for path in candidate_paths:
            if path.exists():
                return path
        
        # If not found, return the first candidate path
        # (will raise error when actually accessed)
        return candidate_paths[0]
    
    def validate(self) -> None:
        """
        Validate that all required paths exist.
        
        Raises:
            FileNotFoundError: If meta_db_path does not exist.
        """
        if not self.meta_db_path.exists():
            raise FileNotFoundError(
                f"DocMeta database not found at: {self.meta_db_path}\n"
                f"Please ensure the database is created at this location."
            )


# Global default config instance
_default_config: Optional[QueryConfig] = None


def get_default_config() -> QueryConfig:
    """
    Get or create the default QueryConfig instance.
    
    This provides a singleton config that can be used across the module.
    """
    global _default_config
    if _default_config is None:
        _default_config = QueryConfig()
    return _default_config


def set_default_config(config: QueryConfig) -> None:
    """
    Set a custom default QueryConfig.
    
    Useful for testing or custom deployments.
    """
    global _default_config
    _default_config = config