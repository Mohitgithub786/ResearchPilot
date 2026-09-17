"""
Retrieval-Augmented Generation orchestration.

Coordinates retrieval, prompt construction, LLM generation, citation
assembly, and message persistence for session-aware conversations.
"""

from app.core.logging import get_logger

from sqlalchemy.orm import Session

from app.schemas.chat import ChatResponse, SourceReference
from app.schemas.retrieve import RetrieveResult
from app.services.llm_service import generate_answer
from app.services.memory_service import (
    format_history,
    get_recent_history,
    save_message,
)
from app.services.prompt_builder import build_conversational_rag_prompt
from app.services.retrieval_service import retrieve_chunks

logger = get_logger(__name__)

NO_CONTEXT_ANSWER = (
    "I could not find sufficient information in the uploaded documents."
)


def _has_meaningful_context(chunks: list[RetrieveResult]) -> bool:
    return bool(chunks) and any(chunk.chunk_text.strip() for chunk in chunks)


def build_citations(chunks: list[RetrieveResult]) -> list[SourceReference]:
    """
    Deduplicate citations by (filename, page_number) while preserving order
    of first appearance in the retrieved ranking.
    """
    seen: set[tuple[str, int | None]] = set()
    sources: list[SourceReference] = []

    for chunk in chunks:
        key = (chunk.source_filename, chunk.page_number)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            SourceReference(
                filename=chunk.source_filename,
                page_number=chunk.page_number,
            )
        )

    return sources


def generate_chat_response(
    db: Session, question: str, session_id: int
) -> ChatResponse:
    """
    Full RAG pipeline: retrieve → history → prompt → generate → persist → cite.

    1. Retrieve relevant chunks from ChromaDB
    2. Fetch recent conversation history for the session
    3. Build a conversational prompt with history + context
    4. Generate an answer via the configured LLM
    5. Persist both user and assistant messages
    6. Return the answer with source citations
    """
    # Step 1: Retrieve
    chunks = retrieve_chunks(db, question)

    if not _has_meaningful_context(chunks):
        # Still persist the exchange even when there is no context
        save_message(db, session_id, "user", question)
        save_message(db, session_id, "assistant", NO_CONTEXT_ANSWER)
        return ChatResponse(answer=NO_CONTEXT_ANSWER, sources=[])

    # Step 2: Fetch recent conversation history
    recent_messages = get_recent_history(db, session_id)
    history_text = format_history(recent_messages)

    # Step 3: Build conversational prompt
    prompt = build_conversational_rag_prompt(question, chunks, history_text)

    # --- Diagnostic logging (Phase 7 investigation) ---
    logger.info("=" * 60)
    logger.info("SESSION ID: %s", session_id)
    logger.info(
        "HISTORY INCLUDED (%d messages): %s",
        len(recent_messages),
        repr(history_text) if history_text else "(empty)",
    )
    for i, c in enumerate(chunks, 1):
        logger.info(
            "RETRIEVED CHUNK [%d]: source=%s, page=%s, text=%.80s...",
            i, c.source_filename, c.page_number, c.chunk_text,
        )
    logger.info("FINAL PROMPT:\n%s", prompt)
    logger.info("=" * 60)

    # Step 4: Generate
    answer = generate_answer(prompt)

    # Step 5: Persist messages
    save_message(db, session_id, "user", question)
    save_message(db, session_id, "assistant", answer)

    # Step 6: Build citations
    sources = build_citations(chunks)

    return ChatResponse(answer=answer, sources=sources)


