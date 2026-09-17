"""Tests for session management, message persistence, and conversational memory."""


from tests.helpers import seed_document_with_chunks


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------


def test_create_session(client):
    test_client, _sf, _cd = client

    response = test_client.post("/sessions")

    assert response.status_code == 201
    body = response.json()
    assert "session_id" in body
    assert body["title"] == "New Chat"


def test_list_sessions(client):
    test_client, _sf, _cd = client

    test_client.post("/sessions")
    test_client.post("/sessions")

    response = test_client.get("/sessions")
    assert response.status_code == 200
    sessions = response.json()
    assert len(sessions) == 2
    # Most recently updated first
    assert sessions[0]["id"] >= sessions[1]["id"]


def test_delete_session(client):
    test_client, _sf, _cd = client

    create_resp = test_client.post("/sessions")
    session_id = create_resp.json()["session_id"]

    delete_resp = test_client.delete(f"/sessions/{session_id}")
    assert delete_resp.status_code == 204

    # Session should be gone
    list_resp = test_client.get("/sessions")
    session_ids = [s["id"] for s in list_resp.json()]
    assert session_id not in session_ids


def test_delete_session_cascades_messages(client):
    test_client, session_factory, _cd = client
    seed_document_with_chunks(
        session_factory,
        "attention.pdf",
        ["Self-attention computes weighted relationships between tokens."],
    )

    create_resp = test_client.post("/sessions")
    session_id = create_resp.json()["session_id"]

    # Create some messages via chat
    test_client.post(
        "/chat",
        json={"session_id": session_id, "question": "What is self-attention?"},
    )

    # Verify messages exist
    msgs_resp = test_client.get(f"/sessions/{session_id}/messages")
    assert len(msgs_resp.json()) > 0

    # Delete session
    test_client.delete(f"/sessions/{session_id}")

    # Messages endpoint should 404 for the deleted session
    msgs_resp = test_client.get(f"/sessions/{session_id}/messages")
    assert msgs_resp.status_code == 404


def test_delete_nonexistent_session(client):
    test_client, _sf, _cd = client

    response = test_client.delete("/sessions/9999")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Message persistence
# ---------------------------------------------------------------------------


def test_message_persistence(client):
    test_client, session_factory, _cd = client
    seed_document_with_chunks(
        session_factory,
        "attention.pdf",
        ["Self-attention computes weighted relationships between tokens."],
    )

    create_resp = test_client.post("/sessions")
    session_id = create_resp.json()["session_id"]

    test_client.post(
        "/chat",
        json={"session_id": session_id, "question": "What is self-attention?"},
    )

    msgs_resp = test_client.get(f"/sessions/{session_id}/messages")
    assert msgs_resp.status_code == 200
    messages = msgs_resp.json()

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is self-attention?"
    assert messages[1]["role"] == "assistant"
    assert len(messages[1]["content"]) > 0


def test_get_messages_nonexistent_session(client):
    test_client, _sf, _cd = client

    response = test_client.get("/sessions/9999/messages")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Conversational continuity — multi-turn pronoun resolution
# ---------------------------------------------------------------------------


def test_conversational_continuity(client):
    """The assistant should understand 'its' refers to self-attention."""
    test_client, session_factory, _cd = client
    seed_document_with_chunks(
        session_factory,
        "attention.pdf",
        [
            "Self-attention computes weighted relationships between tokens.",
            "The limitations of self-attention include quadratic complexity.",
        ],
    )

    create_resp = test_client.post("/sessions")
    session_id = create_resp.json()["session_id"]

    # Turn 1
    r1 = test_client.post(
        "/chat",
        json={"session_id": session_id, "question": "What is self-attention?"},
    )
    assert r1.status_code == 200

    # Turn 2 — follow-up with pronoun reference
    r2 = test_client.post(
        "/chat",
        json={"session_id": session_id, "question": "What are its limitations?"},
    )
    assert r2.status_code == 200
    body = r2.json()
    # The test LLM stub recognises conversation history + "limitation" keyword
    assert "limitation" in body["answer"].lower()


# ---------------------------------------------------------------------------
# Memory window truncation
# ---------------------------------------------------------------------------


def test_memory_window_truncation(client, monkeypatch):
    """Only the last N exchanges should appear in the prompt history."""
    test_client, session_factory, _cd = client

    # Set a very small window
    monkeypatch.setattr("app.core.config.MEMORY_WINDOW", 1)
    monkeypatch.setattr("app.services.memory_service.MEMORY_WINDOW", 1)

    seed_document_with_chunks(
        session_factory,
        "attention.pdf",
        ["Self-attention computes weighted relationships between tokens."],
    )

    create_resp = test_client.post("/sessions")
    session_id = create_resp.json()["session_id"]

    # Create 3 exchanges
    for q in [
        "What is self-attention?",
        "Explain positional encoding.",
        "What are transformers?",
    ]:
        test_client.post(
            "/chat", json={"session_id": session_id, "question": q}
        )

    # Verify all 6 messages are persisted
    msgs_resp = test_client.get(f"/sessions/{session_id}/messages")
    all_messages = msgs_resp.json()
    assert len(all_messages) == 6

    # But the memory window should only include the last 1 exchange (2 messages)
    from app.database.session import SessionLocal

    db = SessionLocal()
    try:
        # This won't use the test DB directly, so let's test via the
        # session_factory from conftest
        pass
    finally:
        db.close()

    # Use the test session factory to validate the window
    from app.services.memory_service import get_recent_history as grh

    db = session_factory()
    try:
        recent = grh(db, session_id, window=1)
        assert len(recent) == 2  # 1 pair = 2 messages
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Multi-session isolation
# ---------------------------------------------------------------------------


def test_multi_session_isolation(client):
    """Messages from session A must not appear in session B."""
    test_client, session_factory, _cd = client
    seed_document_with_chunks(
        session_factory,
        "attention.pdf",
        ["Self-attention computes weighted relationships between tokens."],
    )

    # Create two sessions
    s1 = test_client.post("/sessions").json()["session_id"]
    s2 = test_client.post("/sessions").json()["session_id"]

    # Chat in session 1
    test_client.post(
        "/chat",
        json={"session_id": s1, "question": "What is self-attention?"},
    )

    # Chat in session 2
    test_client.post(
        "/chat",
        json={"session_id": s2, "question": "Explain transformers."},
    )

    # Session 1 messages
    msgs_s1 = test_client.get(f"/sessions/{s1}/messages").json()
    assert len(msgs_s1) == 2
    assert msgs_s1[0]["content"] == "What is self-attention?"

    # Session 2 messages — should NOT contain session 1 content
    msgs_s2 = test_client.get(f"/sessions/{s2}/messages").json()
    assert len(msgs_s2) == 2
    assert msgs_s2[0]["content"] == "Explain transformers."

    # Cross-check: no leakage
    s2_contents = [m["content"] for m in msgs_s2]
    assert "What is self-attention?" not in s2_contents


# ---------------------------------------------------------------------------
# Persistence across requests (simulates restart within a test)
# ---------------------------------------------------------------------------


def test_messages_survive_across_requests(client):
    """Messages should be readable in subsequent requests (SQLite persistence)."""
    test_client, session_factory, _cd = client
    seed_document_with_chunks(
        session_factory,
        "attention.pdf",
        ["Self-attention computes weighted relationships between tokens."],
    )

    create_resp = test_client.post("/sessions")
    session_id = create_resp.json()["session_id"]

    test_client.post(
        "/chat",
        json={"session_id": session_id, "question": "What is self-attention?"},
    )

    # Read back in a fresh request
    msgs = test_client.get(f"/sessions/{session_id}/messages").json()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
