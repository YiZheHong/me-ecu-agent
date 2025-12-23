from pathlib import Path
import sqlite3
import json
import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from me_ecu_agent.data_schema import DocMeta
from .config import IngestConfig
from .util import (
    extract_doc_meta,
    build_documents,
)

# Initialize logger
logger = logging.getLogger(__name__)


# -----------------------------
# Embedding factory
# -----------------------------

def get_embeddings(config: Optional[IngestConfig] = None) -> HuggingFaceEmbeddings:
    """
    Create and return the embedding model used for vector indexing.
    
    Args:
        config: IngestConfig instance. If None, uses default config.
    
    Returns:
        HuggingFaceEmbeddings instance
    """
    if config is None:
        raise ValueError("Configuration must be provided")
    
    logger.debug(f"Initializing embeddings with model: {config.embedding_model}")
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model
    )


# -----------------------------
# SQLite: DocMeta persistence
# -----------------------------

def init_meta_db(db_path: Path) -> None:
    """
    Initialize the SQLite database for storing DocMeta.
    The table is created if it does not already exist.
    
    Args:
        db_path: Path to SQLite database file
    """
    logger.debug(f"Initializing metadata database: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_meta (
            doc_uid TEXT PRIMARY KEY,
            source_filename TEXT,
            product_line TEXT,
            series TEXT,
            model_type TEXT,
            covered_models TEXT,          -- JSON array
            model_inherits_from TEXT,
            status TEXT
        )
        """
    )

    conn.commit()
    conn.close()
    logger.debug("Metadata database initialized successfully")


def save_doc_metas(metas: List[DocMeta], db_path: Path) -> None:
    """
    Persist a list of DocMeta objects into SQLite.

    Existing rows with the same doc_uid will be replaced.
    
    Args:
        metas: List of DocMeta objects to save
        db_path: Path to SQLite database file
    """
    logger.info(f"Saving {len(metas)} document metadata records to database")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for m in metas:
        cur.execute(
            """
            INSERT OR REPLACE INTO doc_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                m.doc_uid,
                m.source_filename,
                m.product_line,
                m.series,
                m.model_type,
                json.dumps(m.covered_models),
                m.model_inherits_from,
                m.status,
            ),
        )
        logger.debug(f"Saved metadata for: {m.source_filename} (doc_uid: {m.doc_uid[:8]}...)")

    conn.commit()
    conn.close()
    logger.info(f"Successfully saved metadata to {db_path}")


# -----------------------------
# Vectorstore build & persist
# -----------------------------

def build_and_persist_vectorstore(
    documents: List[Document],
    persist_dir: Path,
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Build a FAISS vectorstore from documents and persist it to disk.
    
    Args:
        documents: List of Document objects to index
        persist_dir: Directory to save the vectorstore
        embeddings: Embedding model to use
    
    Returns:
        FAISS vectorstore instance
    """
    logger.info(f"Building vectorstore from {len(documents)} documents")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    logger.info(f"Persisting vectorstore to {persist_dir}")
    vectorstore.save_local(str(persist_dir))
    logger.info("Vectorstore successfully persisted")
    
    return vectorstore


# -----------------------------
# Main ingest entrypoint
# -----------------------------

def ingest(
    rebuild: bool = False,
    config: IngestConfig = None,
) -> FAISS:
    """
    Ingest markdown documents into:
    1. FAISS vectorstore (chunk-level embeddings)
    2. SQLite database (document-level DocMeta)

    Args:
        rebuild: If True, rebuild from raw markdown. If False, load from disk.
        config: IngestConfig instance.
    
    Returns:
        FAISS vectorstore instance
        
    Example:
        >>> # Use default configuration
        >>> vectorstore = ingest(rebuild=True)
        
        >>> # Use custom configuration
        >>> from config import IngestConfig
        >>> custom_config = IngestConfig(
        ...     embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ...     chunk_size=500,
        ...     chunk_overlap=50
        ... )
        >>> vectorstore = ingest(rebuild=True, config=custom_config)
        
        >>> # Use configuration from dict
        >>> config_dict = {
        ...     "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        ...     "chunk_size": 500,
        ...     "chunk_overlap": 50,
        ... }
        >>> custom_config = IngestConfig.from_dict(config_dict)
        >>> vectorstore = ingest(rebuild=True, config=custom_config)
    """
    # Use default config if not provided
    if config is None:
        raise ValueError("Configuration must be provided")
    
    logger.info("="*80)
    logger.info("INGESTION CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Configuration:\n{config}")

    # Ensure directories exist
    config.meta_dir.mkdir(parents=True, exist_ok=True)
    config.vector_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directories exist: meta_dir={config.meta_dir}, vector_dir={config.vector_dir}")
    
    # Get embeddings
    embeddings = get_embeddings(config)

    if rebuild:
        # -----------------------------
        # Rebuild from raw markdown
        # -----------------------------
        logger.info("="*80)
        logger.info("REBUILDING FROM RAW MARKDOWN")
        logger.info("="*80)
        logger.info(f"Data directory: {config.data_dir}")
        logger.info(f"Vector directory: {config.vector_dir}")
        logger.info(f"Meta database: {config.meta_db_path}")
        
        all_documents: List[Document] = []
        all_metas: List[DocMeta] = []

        # Initialize metadata database
        init_meta_db(config.meta_db_path)

        # Process each markdown file
        md_files = list(config.data_dir.rglob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files to process")
        
        for i, md_path in enumerate(md_files, 1):
            logger.info(f"[{i}/{len(md_files)}] Processing: {md_path.name}")
            
            # Extract document-level metadata
            meta = extract_doc_meta(md_path)
            all_metas.append(meta)

            # Build chunk-level documents with config
            docs = build_documents(
                md_path, 
                meta,
                max_chars=config.chunk_size,
                overlap_chars=config.chunk_overlap,
            )
            all_documents.extend(docs)
            logger.info(f"  → Generated {len(docs)} chunks")

        logger.info(f"Total chunks generated: {len(all_documents)}")
        logger.info(f"Total documents processed: {len(all_metas)}")
        
        logger.info("Building and persisting vectorstore...")
        vectorstore = build_and_persist_vectorstore(
            all_documents,
            config.vector_dir,
            embeddings
        )
        logger.info(f"✓ Vectorstore saved to {config.vector_dir}")

        # Persist DocMeta to SQLite
        save_doc_metas(all_metas, config.meta_db_path)
        logger.info(f"✓ Metadata saved to {config.meta_db_path}")

    else:
        # -----------------------------
        # Load existing artifacts
        # -----------------------------
        logger.info("="*80)
        logger.info("LOADING EXISTING VECTORSTORE")
        logger.info("="*80)
        logger.info(f"Vector directory: {config.vector_dir}")
        
        vectorstore = FAISS.load_local(
            str(config.vector_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"✓ Vectorstore loaded from {config.vector_dir}")

    logger.info("="*80)
    logger.info("INGESTION COMPLETE")
    logger.info("="*80)

    return vectorstore