"""
ME ECU Agent - Ingestion Module

This module provides document ingestion and chunking capabilities for the ECU RAG system.

Main components:
- IngestConfig: Configuration management for ingestion
- ingest: Main ingestion function with flexible configuration
- ingest_from_dict: Convenience function for dict-based configuration
"""

from me_ecu_agent.ingest.config import IngestConfig
from me_ecu_agent.ingest.ingest import ingest

__version__ = "1.0.0"

__all__ = [
    "IngestConfig",
    "ingest",
]