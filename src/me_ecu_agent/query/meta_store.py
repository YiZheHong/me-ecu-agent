"""
MetaStore: Centralized loading and caching of document metadata.

This module provides a cached interface to the doc_meta SQLite database,
avoiding repeated database reads during query operations.
The MetaStore is implemented as a singleton class - all instantiations
return the same instance.
"""
from typing import List, Dict
from pathlib import Path
import sqlite3
import json

from me_ecu_agent.data_schema import DocMeta
from me_ecu_agent.query.config import QueryConfig


class MetaStore:
    """
    A cached store for document metadata with singleton pattern.
    
    All instantiations return the same instance. Data is loaded from SQLite
    automatically on first instantiation and cached in memory for fast access.
    """
    
    _instance = None
    
    def __new__(cls, config: QueryConfig = None):
        """
        Ensure only one MetaStore instance exists across the application.
        
        Args:
            config: QueryConfig instance. Only used on first instantiation.
        
        Returns:
            The singleton MetaStore instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: QueryConfig = None):
        """
        Initialize the MetaStore. Only performs actual initialization on first call.
        Subsequent calls are no-ops.
        
        Args:
            config: QueryConfig instance. If None, uses default config.
        """
        # Only initialize once, even if __init__ is called multiple times
        if not hasattr(self, '_config'):
            self.config = config
            self._metas: List[DocMeta] = []
            self._metas_by_uid: Dict[str, DocMeta] = {}
            self._loaded = False
            # Automatically load data on first instantiation
            self.load()
    
    def load(self) -> None:
        """
        Load all DocMeta from the database into memory.
        
        This is called automatically during initialization.
        Calling it again will skip if data is already loaded.
        """
        # Skip if data is already loaded
        if self._loaded:
            return
        
        db_path = self.config.meta_dir / "doc_meta.sqlite"
        
        if not db_path.exists():
            raise FileNotFoundError(f"DocMeta database not found: {db_path}")
        
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Fetch all metadata from the database
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
        
        # Parse rows into DocMeta objects and build lookup structures
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
            RuntimeError: If data loading failed.
        """
        if not self._loaded:
            raise RuntimeError("MetaStore failed to load data.")
        return self._metas
    
    def is_loaded(self) -> bool:
        """
        Check if the metadata has been loaded from the database.
        
        Returns:
            True if data is loaded, False otherwise.
        """
        return self._loaded

def reset_meta_store() -> None:
    """
    Reset the global MetaStore instance.
    
    This removes the singleton instance, allowing a new one to be created
    with potentially different configuration. Useful for testing or when
    database content changes.
    """
    MetaStore._instance = None