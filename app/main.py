from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import validate_config
from app.core.errors import (
    ResearchPilotError,
    DocumentNotFoundError,
    RetrievalError,
    EmbeddingError,
    ResearchExecutionError,
    ConfigurationError
)
from app.core.logging import get_logger
from app.database.session import init_db, SessionLocal
from app.routers import chat, documents, hello, research, retrieve, sessions
from app.services.embedding_service import get_embedding_model

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ResearchPilot validation...")
    validate_config()
    init_db()
    
    # Eagerly load the embedding model so the first request isn't slow.
    # The @lru_cache inside get_embedding_model() ensures this runs once.
    try:
        get_embedding_model()
        logger.info("Embedding model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load embedding model on startup: {e}")
        
    logger.info("ResearchPilot startup complete.")
    yield


app = FastAPI(title="ResearchPilot", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(ResearchPilotError)
async def researchpilot_error_handler(request: Request, exc: ResearchPilotError):
    logger.error(f"ResearchPilotError: {exc.message}")
    status_code = 500
    if isinstance(exc, DocumentNotFoundError):
        status_code = 404
    elif isinstance(exc, ConfigurationError):
        status_code = 500
    elif isinstance(exc, (RetrievalError, EmbeddingError, ResearchExecutionError)):
        status_code = 500
        
    return JSONResponse(
        status_code=status_code,
        content={"error": exc.message},
    )

app.include_router(hello.router)
app.include_router(documents.router)
app.include_router(sessions.router)
app.include_router(retrieve.router)
app.include_router(chat.router)
app.include_router(research.router)
@app.get("/health")
def health_check():
    health_status = {
        "status": "healthy",
        "database": "disconnected",
        "chroma": "disconnected",
        "embedding_provider": "unavailable"
    }
    
    # Check Database
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        health_status["database"] = "connected"
    except Exception as e:
        logger.error(f"Health check failed for database: {e}")
        health_status["status"] = "unhealthy"
    finally:
        db.close()
        
    # Check Chroma
    try:
        from app.services.chroma_service import get_chroma_client
        client = get_chroma_client()
        client.heartbeat()
        health_status["chroma"] = "connected"
    except Exception as e:
        logger.error(f"Health check failed for chroma: {e}")
        health_status["status"] = "unhealthy"
        
    # Check Embedding Provider
    try:
        get_embedding_model()
        health_status["embedding_provider"] = "available"
    except Exception as e:
        logger.error(f"Health check failed for embedding provider: {e}")
        health_status["status"] = "unhealthy"
        
    return health_status

import gradio as gr
from gradio_app.app import demo
app = gr.mount_gradio_app(app, demo, path="/")
