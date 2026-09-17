"""
Conversational memory management.

Handles session lifecycle, message persistence, and history retrieval
with a configurable sliding-window strategy.
"""

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import MEMORY_WINDOW
from app.models.chat_session import ChatSession
from app.models.message import Message


def create_session(db: Session) -> ChatSession:
    """Create a new chat session with the default title."""
    chat_session = ChatSession()
    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)
    return chat_session


def list_sessions(db: Session) -> list[ChatSession]:
    """Return all sessions ordered by most recently updated first."""
    return (
        db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
    )


def get_session(db: Session, session_id: int) -> ChatSession:
    """Fetch a session by ID or raise HTTP 404."""
    chat_session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if chat_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return chat_session


def delete_session(db: Session, session_id: int) -> None:
    """Delete a session and cascade-delete its messages."""
    chat_session = get_session(db, session_id)
    db.delete(chat_session)
    db.commit()


def get_session_messages(db: Session, session_id: int) -> list[Message]:
    """Return all messages for a session in chronological order."""
    get_session(db, session_id)  # validates session exists
    return (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .all()
    )


def get_recent_history(
    db: Session, session_id: int, window: int | None = None
) -> list[Message]:
    """
    Fetch the last N user-assistant exchange *pairs* for a session.

    A window of 5 means the last 5 pairs (10 messages). Messages are
    returned in chronological order so they can be fed straight into the
    prompt builder.
    """
    limit = (window or MEMORY_WINDOW) * 2  # each exchange = 2 messages
    recent = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(recent))


def format_history(messages: list[Message]) -> str:
    """
    Format a list of messages into a prompt-ready conversation block.

    Example output:
        User: What is self-attention?
        Assistant: Self-attention computes weighted relationships...
    """
    if not messages:
        return ""
    lines: list[str] = []
    for msg in messages:
        role_label = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role_label}: {msg.content}")
    return "\n\n".join(lines)


def save_message(
    db: Session, session_id: int, role: str, content: str
) -> Message:
    """Persist a single message and touch the session's updated_at."""
    message = Message(session_id=session_id, role=role, content=content)
    db.add(message)

    # Touch the parent session so updated_at reflects latest activity
    chat_session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if chat_session is not None:
        from datetime import datetime

        chat_session.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(message)
    return message
