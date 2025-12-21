import os
import re
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from langchain_core.documents import Document
from me_ecu_agent.data_schema import DocMeta

# ============================================================
# Utility functions - Metadata extraction
# ============================================================

def sha256_text(text: str) -> str:
    """Generate a SHA-256 hash from normalized text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def normalize_text(text: str) -> str:
    """
    Normalize text for stable hashing:
    - Normalize line endings
    - Collapse consecutive whitespace
    """
    text = text.replace("\r\n", "\n")
    return re.sub(r"[ \t]+", " ", text).strip()

def parse_filename(filename: str) -> Tuple[str, str, str]:
    """
    Parse structured metadata from filename.

    Returns:
        product_line, series, model_type
    """
    name = Path(filename).stem  # remove .md
    parts = name.split("_")
    product_series = parts[0]
    product_line = product_series.split("-")[0]
    series = product_series

    # Manual / Base / Plus
    tail = parts[-1]

    if tail.lower() == "manual":
        model_type = "Base"
    else:
        model_type = tail  # Base / Plus

    return product_line, series, model_type

def resolve_model_inheritance(
    series: str,
    model_type: Optional[str],
) -> Optional[str]:
    """
    Resolve model inheritance relationship.
    """
    if model_type == "Plus":
        # Plus always inherits from Base of the same series
        return f"{series}-Base"

    return None

def extract_covered_models(text: str, product_line: str) -> List[str]:
    """
    Extract covered models strictly matching:
    <product_line>-<digits + optional letters>

    Rules:
    - No alphanumeric or underscore prefix
    - No alphanumeric suffix
    - Product line is already known from filename
    """
    pattern = re.compile(
    rf"\*\*({re.escape(product_line)}-\d+[A-Za-z]*)\*\*"
    )

    models = sorted({m.group(1) for m in pattern.finditer(text)})
    return models

def extract_doc_meta(md_path: Path) -> DocMeta:
    raw_text = md_path.read_text(encoding="utf-8", errors="ignore")
    doc_uid = sha256_text(raw_text)

    product_line, series, model_type = parse_filename(md_path.name)

    covered_models = extract_covered_models(raw_text, product_line)
    model_inherits_from = resolve_model_inheritance(series, model_type)

    status = "legacy" if "legacy" in raw_text.lower() else "online"
    print(DocMeta(
        doc_uid=doc_uid,
        source_filename=md_path.name,
        product_line=product_line,
        series=series,
        model_type=model_type,
        covered_models=covered_models,
        model_inherits_from=model_inherits_from,
        status=status,
    ))
    return DocMeta(
        doc_uid=doc_uid,
        source_filename=md_path.name,
        product_line=product_line,
        series=series,
        model_type=model_type,
        covered_models=covered_models,
        model_inherits_from=model_inherits_from,
        status=status,
    )

# ============================================================
# Utility functions - Vector store document building
# ============================================================
def build_documents(md_path: Path, meta: DocMeta) -> list[Document]:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    sections = chunk_by_h2(text)

    documents = []
    for sec in sections:
        documents.append(
            Document(
                page_content=sec["content"],
                metadata={
                    "doc_uid": meta.doc_uid,
                    "source_filename": meta.source_filename,
                    "product_line": meta.product_line,
                    "series": meta.series,
                    "model_type": meta.model_type,
                    "covered_models": meta.covered_models,
                    "model_inherits_from": meta.model_inherits_from,
                    "status": meta.status,
                    "section_title": sec["section_title"],
                    "section_path": sec["section_path"],
                    "chunk_index": sec["chunk_index"],
                },
            )
        )
    return documents

def chunk_by_h2(
    text: str,
    max_chars: int = 1500,
    overlap_chars: int = 300,
) -> List[Dict[str, Any]]:
    """
    Chunk markdown text by H2 (##) headings.

    Rules:
    - H1 (#) is treated as document-level context (not a chunk)
    - Each H2 (##) defines a primary semantic section
    - H3+ stays inside its parent H2
    - Long H2 sections are split with sliding window + overlap
    - No chunk crosses H2 boundaries

    Returns:
        List of dicts:
        {
            "content": str,
            "section_title": str,
            "section_path": [h1_title, h2_title],
            "chunk_index": int,
        }
    """
    H1_PATTERN = re.compile(r"^# (.+)", re.MULTILINE)
    H2_PATTERN = re.compile(r"^## (.+)", re.MULTILINE)
    chunks: List[Dict[str, Any]] = []

    # -------------------------------
    # Extract H1 (document title)
    # -------------------------------
    h1_match = H1_PATTERN.search(text)
    h1_title = h1_match.group(1).strip() if h1_match else None

    # -------------------------------
    # Find all H2 sections
    # -------------------------------
    h2_matches = list(H2_PATTERN.finditer(text))

    # If no H2 exists, treat whole doc as one section
    if not h2_matches:
        h2_matches = [None]

    for idx, h2_match in enumerate(h2_matches):
        # Determine section boundaries
        if h2_match:
            section_start = h2_match.end()
            section_title = h2_match.group(1).strip()
        else:
            section_start = 0
            section_title = "Document"

        if idx + 1 < len(h2_matches) and h2_matches[idx + 1]:
            section_end = h2_matches[idx + 1].start()
        else:
            section_end = len(text)

        section_text = text[section_start:section_end].strip()
        if not section_text:
            continue

        # --------------------------------
        # Split long sections with overlap
        # --------------------------------
        if len(section_text) <= max_chars:
            chunks.append(
                {
                    "content": section_text,
                    "section_title": section_title,
                    "section_path": [h1_title, section_title] if h1_title else [section_title],
                    "chunk_index": 0,
                }
            )
        else:
            start = 0
            chunk_idx = 0
            while start < len(section_text):
                end = start + max_chars
                chunk_text = section_text[start:end].strip()

                chunks.append(
                    {
                        "content": chunk_text,
                        "section_title": section_title,
                        "section_path": [h1_title, section_title] if h1_title else [section_title],
                        "chunk_index": chunk_idx,
                    }
                )

                chunk_idx += 1
                start = end - overlap_chars

    return chunks