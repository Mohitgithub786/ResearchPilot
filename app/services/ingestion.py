"""
PDF ingestion service using LangChain document loaders and text splitters.

LangChain Document objects are the standard interchange format for RAG pipelines.
Each Document pairs page_content (text) with metadata (source, page, etc.). Loaders,
splitters, embedders, and retrievers all operate on Documents, so storing chunks
derived from them keeps ResearchPilot aligned with downstream Phase 5 embedding and
retrieval work without re-parsing PDFs.
"""

from pathlib import Path

from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from app.core.config import CHUNK_OVERLAP, CHUNK_SIZE, UPLOAD_DIR
from app.core.logging import get_logger
from app.models.chunk import Chunk
from app.models.document import Document
from app.services.chroma_service import index_chunks

logger = get_logger(__name__)


def _safe_filename(filename: str) -> str:
    return Path(filename).name


def save_upload_file(document_id: int, file: UploadFile) -> Path:
    """Persist an uploaded PDF under data/uploads/."""
    safe_name = _safe_filename(file.filename or "upload.pdf")
    destination = UPLOAD_DIR / f"{document_id}_{safe_name}"
    content = file.file.read()
    destination.write_bytes(content)
    return destination


def load_pdf_documents(file_path: Path, source_filename: str) -> list[LangChainDocument]:
    """
    Extract LangChain Documents from a PDF using PyPDFLoader.

    PyPDFLoader produces one Document per page with metadata including:
    - source: absolute path to the file on disk
    - page: zero-based page index

    We add source_filename so chunks remain traceable to the original upload
    even if files are renamed on disk.
    """
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()
    for doc in documents:
        doc.metadata["source_filename"] = source_filename
    return documents


def split_documents(
    documents: list[LangChainDocument],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[LangChainDocument]:
    """
    Split page-level Documents into smaller chunks with RecursiveCharacterTextSplitter.

    RecursiveCharacterTextSplitter tries separators in order (paragraphs, sentences,
    words) to keep semantically coherent fragments. chunk_overlap preserves context
    at chunk boundaries, which improves retrieval quality in later RAG phases.

    Metadata (source, page, source_filename) is propagated to every resulting chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def _page_number_from_metadata(metadata: dict) -> int | None:
    page = metadata.get("page")
    if page is None:
        return None
    return int(page) + 1  # PyPDFLoader uses zero-based pages; store as 1-based


def persist_chunks(
    db: Session, document_id: int, chunks: list[LangChainDocument]
) -> list[Chunk]:
    """Map LangChain Documents to Chunk ORM rows and persist them."""
    chunk_rows: list[Chunk] = []
    for order, chunk in enumerate(chunks):
        row = Chunk(
            document_id=document_id,
            chunk_text=chunk.page_content,
            page_number=_page_number_from_metadata(chunk.metadata),
            chunk_order=order,
        )
        db.add(row)
        chunk_rows.append(row)
    return chunk_rows


def ingest_pdf(db: Session, file: UploadFile) -> tuple[Document, int]:
    """
    Full ingestion pipeline: save file → load → split → persist chunks → embed.

    Chunk text lives in SQLite; vectors live in ChromaDB keyed by chunk id.
    LLM answer generation is intentionally deferred to Phase 6.
    """
    logger.info("Document upload started for file: %s", file.filename)
    filename = _safe_filename(file.filename or "upload.pdf")
    document = Document(filename=filename)
    db.add(document)
    db.flush()

    file_path = save_upload_file(document.id, file)
    page_documents = load_pdf_documents(file_path, source_filename=filename)
    text_chunks = split_documents(page_documents)
    chunk_rows = persist_chunks(db, document.id, text_chunks)
    db.flush()
    
    logger.info("Chunks generated: %d chunks for document %s", len(chunk_rows), filename)

    index_chunks(chunk_rows, source_filename=filename)

    db.commit()
    db.refresh(document)
    logger.info("Document uploaded successfully: %s (ID: %d)", filename, document.id)
    return document, len(chunk_rows)
