from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.session import MessageResponse, SessionCreateResponse, SessionResponse
from app.services.memory_service import (
    create_session,
    delete_session,
    get_session_messages,
    list_sessions,
)

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.post("", response_model=SessionCreateResponse, status_code=201)
def create_new_session(db: Session = Depends(get_db)) -> SessionCreateResponse:
    chat_session = create_session(db)
    return SessionCreateResponse(session_id=chat_session.id, title=chat_session.title)


@router.get("", response_model=list[SessionResponse])
def list_all_sessions(db: Session = Depends(get_db)) -> list[SessionResponse]:
    return list_sessions(db)


@router.get("/{session_id}/messages", response_model=list[MessageResponse])
def get_messages(
    session_id: int, db: Session = Depends(get_db)
) -> list[MessageResponse]:
    return get_session_messages(db, session_id)


@router.delete("/{session_id}", status_code=204)
def delete_existing_session(session_id: int, db: Session = Depends(get_db)) -> None:
    delete_session(db, session_id)
