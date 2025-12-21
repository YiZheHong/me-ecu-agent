from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from .util import (
    extract_doc_meta,
    build_documents
)

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def build_and_persist_vectorstore(
    documents: list[Document],
    persist_dir: Path,
):
    vectorstore = FAISS.from_documents(documents, get_embeddings())
    vectorstore.save_local(str(persist_dir))

def ingest(rebuild: bool = False) -> FAISS:
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data"
    persist_dir = project_root/ "src" /"me_ecu_agent"/ "rag"

    embeddings = get_embeddings()

    if rebuild:
        all_documents = []
        for md_path in data_dir.rglob("*.md"):
            meta = extract_doc_meta(md_path)
            docs = build_documents(md_path, meta)
            all_documents.extend(docs)

        vectorstore = FAISS.from_documents(all_documents, embeddings)
        vectorstore.save_local(str(persist_dir))
    else:
        vectorstore = FAISS.load_local(
            str(persist_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    return vectorstore




