"""
MetaStore: Centralized loading and caching of document metadata.

This module provides a cached interface to the doc_meta SQLite database,
avoiding repeated database reads during query operations.
"""
from typing import List, Dict
from pathlib import Path
import sqlite3
import json

from me_ecu_agent.data_schema import DocMeta
from me_ecu_agent.query.config import QueryConfig, get_default_config


class MetaStore:
    """
    A cached store for document metadata.
    
    Loads DocMeta from SQLite once and provides fast in-memory access.
    This class is designed to be instantiated once at application startup.
    """
    
    def __init__(self, config: QueryConfig = None):
        """
        Args:
            config: QueryConfig instance. If None, uses default config.
        """
        self.config = config or get_default_config()
        self._metas: List[DocMeta] = []
        self._metas_by_uid: Dict[str, DocMeta] = {}
        self._loaded = False
    
    def load(self) -> None:
        """
        Load all DocMeta from the database into memory.
        
        This should be called once during initialization.
        Subsequent calls will reload the data.
        """
        db_path = self.config.meta_db_path
        
        if not db_path.exists():
            raise FileNotFoundError(f"DocMeta database not found: {db_path}")
        
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        rows = cur.execute(
            """
            SELECT
                doc_uid,
                source_filename,
                product_line,
                series,
                model_type,
                covered_models,
                model_inherits_from,
                status
            FROM doc_meta
            """
        ).fetchall()
        
        conn.close()
        
        # Parse rows into DocMeta objects
        self._metas = []
        self._metas_by_uid = {}
        
        for r in rows:
            meta = DocMeta(
                doc_uid=r[0],
                source_filename=r[1],
                product_line=r[2],
                series=r[3],
                model_type=r[4],
                covered_models=json.loads(r[5]),
                model_inherits_from=r[6],
                status=r[7],
            )
            self._metas.append(meta)
            self._metas_by_uid[meta.doc_uid] = meta
        
        self._loaded = True
    
    def get_all(self) -> List[DocMeta]:
        """
        Get all DocMeta objects.
        
        Returns:
            List of all DocMeta objects.
            
        Raises:
            RuntimeError: If load() has not been called.
        """
        if not self._loaded:
            raise RuntimeError("MetaStore not loaded. Call load() first.")
        return self._metas
    
    def get_by_uid(self, doc_uid: str) -> DocMeta:
        """
        Get a specific DocMeta by doc_uid.
        
        Args:
            doc_uid: Document unique identifier.
            
        Returns:
            DocMeta object.
            
        Raises:
            KeyError: If doc_uid not found.
            RuntimeError: If load() has not been called.
        """
        if not self._loaded:
            raise RuntimeError("MetaStore not loaded. Call load() first.")
        return self._metas_by_uid[doc_uid]
    
    def is_loaded(self) -> bool:
        """Check if the store has been loaded."""
        return self._loaded


# Global singleton instance
_global_store: MetaStore = None


def get_meta_store(config: QueryConfig = None) -> MetaStore:
    """
    Get or create the global MetaStore instance.
    
    Args:
        config: Optional QueryConfig. Only used on first call.
    
    Returns:
        The global MetaStore instance.
    """
    global _global_store
    if _global_store is None:
        _global_store = MetaStore(config)
        _global_store.load()
    return _global_store


def reset_meta_store() -> None:
    """
    Reset the global MetaStore.
    
    Useful for testing or when database content changes.
    """
    global _global_store
    _global_store = None