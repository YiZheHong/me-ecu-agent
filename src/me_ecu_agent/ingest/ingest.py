from pathlib import Path
import sqlite3
import json
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


def save_doc_metas(metas: List[DocMeta], db_path: Path) -> None:
    """
    Persist a list of DocMeta objects into SQLite.

    Existing rows with the same doc_uid will be replaced.
    
    Args:
        metas: List of DocMeta objects to save
        db_path: Path to SQLite database file
    """
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

    conn.commit()
    conn.close()


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
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(str(persist_dir))
    return vectorstore


# -----------------------------
# Main ingest entrypoint
# -----------------------------

def ingest(
    rebuild: bool = False,
    config: IngestConfig = None,
    verbose: bool = True,
) -> FAISS:
    """
    Ingest markdown documents into:
    1. FAISS vectorstore (chunk-level embeddings)
    2. SQLite database (document-level DocMeta)

    Args:
        rebuild: If True, rebuild from raw markdown. If False, load from disk.
        config: IngestConfig instance.
        verbose: If True, print progress information.
    
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
    
    if verbose:
        print("="*80)
        print("INGESTION CONFIGURATION")
        print("="*80)
        print(config)
        print()

    # Ensure directories exist
    config.meta_dir.mkdir(parents=True, exist_ok=True)
    config.vector_dir.mkdir(parents=True, exist_ok=True)
    
    # Get embeddings
    embeddings = get_embeddings(config)

    if rebuild:
        # -----------------------------
        # Rebuild from raw markdown
        # -----------------------------
        if verbose:
            print("="*80)
            print("REBUILDING FROM RAW MARKDOWN")
            print("="*80)
            print(f"Data directory: {config.data_dir}")
            print(f"Vector directory: {config.vector_dir}")
            print(f"Meta database: {config.meta_db_path}")
            print()
        
        all_documents: List[Document] = []
        all_metas: List[DocMeta] = []

        # Initialize metadata database
        init_meta_db(config.meta_db_path)

        # Process each markdown file
        md_files = list(config.data_dir.rglob("*.md"))
        
        if verbose:
            print(f"Found {len(md_files)} markdown files to process")
            print()
        
        for i, md_path in enumerate(md_files, 1):
            if verbose:
                print(f"[{i}/{len(md_files)}] Processing: {md_path.name}")
            
            # Extract document-level metadata
            meta = extract_doc_meta(md_path, verbose=False)
            all_metas.append(meta)

            # Build chunk-level documents with config
            docs = build_documents(
                md_path, 
                meta,
                max_chars=config.chunk_size,
                overlap_chars=config.chunk_overlap,
                verbose=verbose
            )
            all_documents.extend(docs)
            
            if verbose:
                print(f"  → Generated {len(docs)} chunks")

        if verbose:
            print()
            print(f"Total chunks generated: {len(all_documents)}")
            print(f"Total documents processed: {len(all_metas)}")
            print()
            print("Building and persisting vectorstore...")
        
        vectorstore = build_and_persist_vectorstore(
            all_documents,
            config.vector_dir,
            embeddings
        )
        
        if verbose:
            print(f"✓ Vectorstore saved to {config.vector_dir}")
            print()

        # Persist DocMeta to SQLite
        if verbose:
            print("Saving metadata to database...")
        
        save_doc_metas(all_metas, config.meta_db_path)
        
        if verbose:
            print(f"✓ Metadata saved to {config.meta_db_path}")
            print()

    else:
        # -----------------------------
        # Load existing artifacts
        # -----------------------------
        if verbose:
            print("="*80)
            print("LOADING EXISTING VECTORSTORE")
            print("="*80)
            print(f"Vector directory: {config.vector_dir}")
            print()
        
        vectorstore = FAISS.load_local(
            str(config.vector_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        
        if verbose:
            print(f"✓ Vectorstore loaded from {config.vector_dir}")
            print()

    if verbose:
        print("="*80)
        print("INGESTION COMPLETE")
        print("="*80)
        print()

    return vectorstore