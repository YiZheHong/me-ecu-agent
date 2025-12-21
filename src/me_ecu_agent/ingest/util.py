import os
import re
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, NamedTuple
from langchain_core.documents import Document
from me_ecu_agent.data_schema import DocMeta

# ============================================================
# Data structures for intermediate representations
# ============================================================

class TableMatch(NamedTuple):
    """Represents a matched table in the document"""
    start: int
    end: int
    content: str

class Section(NamedTuple):
    """Represents a document section"""
    start: int
    end: int
    title: str
    section_type: str  # 'h1', 'h2', 'bold', 'default'

class Chunk(NamedTuple):
    """Represents a final chunk"""
    content: str
    section_title: str
    section_path: List[str]
    chunk_index: int
    chunk_type: str  # 'spec' or 'info'

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
    """Extract document-level metadata from markdown file."""
    raw_text = md_path.read_text(encoding="utf-8", errors="ignore")
    doc_uid = sha256_text(raw_text)

    product_line, series, model_type = parse_filename(md_path.name)

    covered_models = extract_covered_models(raw_text, product_line)
    model_inherits_from = resolve_model_inheritance(series, model_type)

    status = "legacy" if "legacy" in raw_text.lower() else "online"
    
    meta = DocMeta(
        doc_uid=doc_uid,
        source_filename=md_path.name,
        product_line=product_line,
        series=series,
        model_type=model_type,
        covered_models=covered_models,
        model_inherits_from=model_inherits_from,
        status=status,
    )
    
    print(meta)
    return meta

# ============================================================
# Layer 1: Content Extraction
# ============================================================

def extract_tables(text: str) -> List[TableMatch]:
    """
    Extract all markdown tables from text.
    
    Returns:
        List of TableMatch objects with start, end, and content.
    """
    TABLE_PATTERN = re.compile(
        r"(\|.+\|[\r\n]+\|[\s\-:|]+\|[\r\n]+(?:\|.+\|[\r\n]*)+)", 
        re.MULTILINE
    )
    
    tables = []
    for match in TABLE_PATTERN.finditer(text):
        tables.append(TableMatch(
            start=match.start(),
            end=match.end(),
            content=match.group(1).strip()
        ))
    
    return tables

def extract_code_blocks(text: str) -> List[TableMatch]:
    """
    Extract code blocks from text (for future use).
    
    Returns:
        List of code block matches (using TableMatch structure for simplicity).
    """
    CODE_PATTERN = re.compile(
        r"```[\w]*\n(.*?)\n```",
        re.MULTILINE | re.DOTALL
    )
    
    blocks = []
    for match in CODE_PATTERN.finditer(text):
        blocks.append(TableMatch(
            start=match.start(),
            end=match.end(),
            content=match.group(0).strip()
        ))
    
    return blocks

# ============================================================
# Layer 2: Section Splitting
# ============================================================

def extract_h1_title(text: str) -> Optional[str]:
    """Extract H1 title from document."""
    H1_PATTERN = re.compile(r"^# (.+)", re.MULTILINE)
    match = H1_PATTERN.search(text)
    return match.group(1).strip() if match else None

def split_by_sections(text: str) -> List[Section]:
    """
    Split text into sections based on H2 and bold numbered headings.
    
    Returns:
        List of Section objects with start, end, title, and type.
    """
    H2_PATTERN = re.compile(r"^## (.+)", re.MULTILINE)
    BOLD_SECTION_PATTERN = re.compile(r"^\*\*(\d+)\.\s*(.+?)\*\*", re.MULTILINE)
    
    sections = []
    
    # Find H2 sections
    for match in H2_PATTERN.finditer(text):
        sections.append(Section(
            start=match.start(),
            end=match.end(),
            title=match.group(1).strip(),
            section_type="h2"
        ))
    
    # Find bold numbered sections
    for match in BOLD_SECTION_PATTERN.finditer(text):
        sections.append(Section(
            start=match.start(),
            end=match.end(),
            title=match.group(2).strip(),
            section_type="bold"
        ))
    
    # Sort by position
    sections.sort(key=lambda x: x.start)
    
    # If no sections found, create a default one
    if not sections:
        sections = [Section(
            start=0,
            end=0,
            title="Document",
            section_type="default"
        )]
    
    return sections

# ============================================================
# Layer 3: Chunk Assembly
# ============================================================

def is_specification_section(section_title: str) -> bool:
    """
    Determine if a section is a specification section based on title.
    
    Args:
        section_title: The section title to check
    
    Returns:
        True if section contains specification keywords
    """
    spec_keywords = [
        'specification',
        'specifications', 
        'specs',
        'technical spec',
        'tech spec',
    ]
    
    title_lower = section_title.lower()
    return any(keyword in title_lower for keyword in spec_keywords)

def create_chunks_from_section(
    section_text: str,
    section_title: str,
    h1_title: Optional[str],
    max_chars: int = 1500,
    overlap_chars: int = 300,
) -> List[Chunk]:
    """
    Create chunks from a single section.
    
    Strategy:
    - If section title contains "specification" keywords → mark as 'spec'
    - Otherwise → mark as 'info'
    - Split long sections with overlap
    
    Args:
        section_text: The text content of the section
        section_title: Title of the section
        h1_title: Document H1 title (for section_path)
        max_chars: Maximum characters per chunk
        overlap_chars: Overlap between chunks for long sections
    
    Returns:
        List of Chunk objects
    """
    chunks = []
    section_path = [h1_title, section_title] if h1_title else [section_title]
    
    # Determine chunk type based on section title
    chunk_type = "spec" if is_specification_section(section_title) else "info"
    
    # Split section into chunks if needed
    if len(section_text) <= max_chars:
        # Single chunk
        chunks.append(Chunk(
            content=section_text,
            section_title=section_title,
            section_path=section_path,
            chunk_index=0,
            chunk_type=chunk_type
        ))
    else:
        # Multiple chunks with overlap
        start = 0
        chunk_idx = 0
        
        while start < len(section_text):
            end = start + max_chars
            chunk_text = section_text[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunks.append(Chunk(
                    content=chunk_text,
                    section_title=section_title,
                    section_path=section_path,
                    chunk_index=chunk_idx,
                    chunk_type=chunk_type
                ))
                chunk_idx += 1
            
            start = end - overlap_chars
    
    return chunks

def chunk_document(
    text: str,
    max_chars: int = 1500,
    overlap_chars: int = 300,
) -> List[Chunk]:
    """
    Main chunking function that orchestrates all layers.
    
    Strategy:
    - Sections with "specification" in title → chunk_type='spec'
    - Other sections → chunk_type='info'
    - Long sections are split with overlap
    
    Args:
        text: Full markdown document text
        max_chars: Maximum characters per chunk
        overlap_chars: Overlap between chunks for long sections
    
    Returns:
        List of Chunk objects
    """
    # Extract H1 title
    h1_title = extract_h1_title(text)
    
    # Split into sections
    sections = split_by_sections(text)
    
    # Process each section
    all_chunks = []
    
    for idx, section in enumerate(sections):
        # Determine section boundaries
        section_start = section.end if section.section_type != "default" else 0
        
        if idx + 1 < len(sections):
            section_end = sections[idx + 1].start
        else:
            section_end = len(text)
        
        section_text = text[section_start:section_end].strip()
        if not section_text:
            continue
        
        # Create chunks for this section
        section_chunks = create_chunks_from_section(
            section_text=section_text,
            section_title=section.title,
            h1_title=h1_title,
            max_chars=max_chars,
            overlap_chars=overlap_chars
        )
        
        all_chunks.extend(section_chunks)
    for c in all_chunks:
        print(c)
        print('-------------------------')
    return all_chunks

# ============================================================
# Document building for vector store
# ============================================================

def build_documents(md_path: Path, meta: DocMeta) -> List[Document]:
    """Build LangChain documents from markdown file."""
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_document(text)

    documents = []
    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk.content,
                metadata={
                    "doc_uid": meta.doc_uid,
                    "source_filename": meta.source_filename,
                    "product_line": meta.product_line,
                    "series": meta.series,
                    "model_type": meta.model_type,
                    "covered_models": meta.covered_models,
                    "model_inherits_from": meta.model_inherits_from,
                    "status": meta.status,
                    "section_title": chunk.section_title,
                    "section_path": chunk.section_path,
                    "chunk_index": chunk.chunk_index,
                    "chunk_type": chunk.chunk_type,
                },
            )
        )
    return documents