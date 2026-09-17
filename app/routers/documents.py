from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.chunk import Chunk
from app.models.document import Document
from app.core.errors import DocumentNotFoundError
from app.schemas.chunk import ChunkResponse
from app.schemas.document import (
    DocumentCreate,
    DocumentResponse,
    DocumentUploadResponse,
)
from app.services.ingestion import ingest_pdf

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("", response_model=DocumentResponse, status_code=201)
def create_document(payload: DocumentCreate, db: Session = Depends(get_db)) -> Document:
    document = Document(filename=payload.filename)
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    document, chunk_count = ingest_pdf(db, file)
    return DocumentUploadResponse(
        id=document.id,
        filename=document.filename,
        uploaded_at=document.uploaded_at,
        chunk_count=chunk_count,
    )


@router.get("", response_model=list[DocumentResponse])
def list_documents(db: Session = Depends(get_db)) -> list[Document]:
    return db.query(Document).order_by(Document.uploaded_at.desc()).all()


@router.get("/{document_id}/chunks", response_model=list[ChunkResponse])
def list_document_chunks(
    document_id: int, db: Session = Depends(get_db)
) -> list[ChunkResponse]:
    document = db.get(Document, document_id)
    if document is None:
        raise DocumentNotFoundError(f"Document with ID {document_id} not found.")

    return (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_order)
        .all()
    )
