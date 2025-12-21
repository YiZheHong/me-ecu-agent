from pathlib import Path
import sqlite3
import json
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from me_ecu_agent.data_schema import DocMeta
from .util import (
    extract_doc_meta,
    build_documents,
)


# -----------------------------
# Embedding factory
# -----------------------------

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Create and return the embedding model used for vector indexing.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------
# SQLite: DocMeta persistence
# -----------------------------

def init_meta_db(db_path: Path) -> None:
    """
    Initialize the SQLite database for storing DocMeta.
    The table is created if it does not already exist.
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
) -> FAISS:
    """
    Build a FAISS vectorstore from documents and persist it to disk.
    """
    vectorstore = FAISS.from_documents(documents, get_embeddings())
    vectorstore.save_local(str(persist_dir))
    return vectorstore


# -----------------------------
# Main ingest entrypoint
# -----------------------------

def ingest(rebuild: bool = False) -> FAISS:
    """
    Ingest markdown documents into:
    1. FAISS vectorstore (chunk-level embeddings)
    2. SQLite database (document-level DocMeta)

    If rebuild=False, both FAISS and SQLite are loaded from disk.
    """
    project_root = Path(__file__).resolve().parents[3]

    data_dir = project_root / "data"
    vector_dir = project_root / "src" / "me_ecu_agent" / "rag"
    meta_dir = project_root / "src" / "me_ecu_agent" / "meta"
    meta_db_path = meta_dir / "doc_meta.sqlite"

    meta_dir.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()

    if rebuild:
        # -----------------------------
        # Rebuild from raw markdown
        # -----------------------------
        all_documents: List[Document] = []
        all_metas: List[DocMeta] = []

        init_meta_db(meta_db_path)

        for md_path in data_dir.rglob("*.md"):
            # Extract document-level metadata
            meta = extract_doc_meta(md_path)
            all_metas.append(meta)

            # Build chunk-level documents
            docs = build_documents(md_path, meta)
            all_documents.extend(docs)
            print('\n\n\n\n')

        # Persist vectorstore
        vectorstore = FAISS.from_documents(all_documents, embeddings)
        vectorstore.save_local(str(vector_dir))

        # Persist DocMeta to SQLite
        save_doc_metas(all_metas, meta_db_path)

    else:
        # -----------------------------
        # Load existing artifacts
        # -----------------------------
        vectorstore = FAISS.load_local(
            str(vector_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    return vectorstore
