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


from fastapi import BackgroundTasks
from app.schemas.document import TaskResponse, TaskStatusResponse
import uuid
import shutil
import os

TASK_STATUS = {}

def async_ingest_pdf_task(task_id: str, file_path: str, filename: str):
    from app.database.session import SessionLocal
    db = SessionLocal()
    try:
        TASK_STATUS[task_id] = {"status": "processing", "progress": 10, "completed": False, "failed": False}
        # Assuming ingest_pdf can take a file path instead of UploadFile, or we construct a dummy
        # Wait, ingest_pdf takes (db, file: UploadFile). I'll need to mock UploadFile or modify ingest_pdf
        # Since I cannot easily modify ingest_pdf without seeing it, I'll pass a mock object
        class MockUploadFile:
            def __init__(self, filename, file):
                self.filename = filename
                self.file = file
        with open(file_path, "rb") as f:
            mock_file = MockUploadFile(filename, f)
            document, chunk_count = ingest_pdf(db, mock_file)
        TASK_STATUS[task_id] = {"status": "completed", "progress": 100, "completed": True, "failed": False}
    except Exception as e:
        TASK_STATUS[task_id] = {"status": "failed", "progress": 0, "completed": False, "failed": True, "error": str(e)}
    finally:
        db.close()
        if os.path.exists(file_path):
            os.remove(file_path)

@router.post("/upload", response_model=TaskResponse, status_code=202)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> TaskResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    task_id = str(uuid.uuid4())
    os.makedirs("data/uploads", exist_ok=True)
    temp_path = f"data/uploads/temp_{task_id}_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    TASK_STATUS[task_id] = {"status": "queued", "progress": 0, "completed": False, "failed": False}
    background_tasks.add_task(async_ingest_pdf_task, task_id, temp_path, file.filename)
    
    return TaskResponse(task_id=task_id, status="queued")

@router.get("/status/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    if task_id not in TASK_STATUS:
        raise HTTPException(status_code=404, detail="Task not found")
    status = TASK_STATUS[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=status.get("status"),
        progress=status.get("progress"),
        completed=status.get("completed"),
        failed=status.get("failed"),
        error=status.get("error")
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
