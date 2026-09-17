from app.services.chroma_service import (
    _resolve_collection_name,
    get_chroma_client,
    reset_chroma_client,
)

from tests.helpers import seed_document_with_chunks


def test_successful_retrieval(client):
    test_client, session_factory, _chroma_dir = client
    seed_document_with_chunks(
        session_factory,
        "attention.pdf",
        [
            "Positional encoding injects order information into transformer inputs.",
            "Self-attention computes weighted relationships between tokens.",
        ],
    )

    response = test_client.post(
        "/retrieve",
        json={"query": "What is positional encoding?"},
    )

    assert response.status_code == 200
    results = response.json()
    assert len(results) >= 1
    top = results[0]
    assert top["source_filename"] == "attention.pdf"
    assert "positional encoding" in top["chunk_text"].lower()
    assert top["similarity_score"] > 0


def test_retrieval_from_multiple_documents(client):
    test_client, session_factory, _chroma_dir = client
    seed_document_with_chunks(
        session_factory,
        "attention.pdf",
        ["Positional encoding is essential for sequence models."],
    )
    seed_document_with_chunks(
        session_factory,
        "database.pdf",
        ["SQLite stores structured relational metadata for documents."],
    )

    response = test_client.post(
        "/retrieve",
        json={"query": "What is positional encoding?"},
    )

    assert response.status_code == 200
    results = response.json()
    assert len(results) >= 1
    assert results[0]["source_filename"] == "attention.pdf"
    assert "positional encoding" in results[0]["chunk_text"].lower()


def test_empty_retrieval_when_no_vectors_indexed(client):
    test_client, _session_factory, _chroma_dir = client

    response = test_client.post(
        "/retrieve",
        json={"query": "What is positional encoding?"},
    )

    assert response.status_code == 200
    assert response.json() == []


def test_chroma_persistence_across_application_restarts(client):
    test_client, session_factory, chroma_dir = client
    document = seed_document_with_chunks(
        session_factory,
        "attention.pdf",
        ["Positional encoding enables transformers to model token order."],
    )

    reset_chroma_client()
    persisted_client = get_chroma_client()
    collection = persisted_client.get_or_create_collection(_resolve_collection_name())
    assert collection.count() >= 1

    response = test_client.post(
        "/retrieve",
        json={"query": "positional encoding transformers"},
    )

    assert response.status_code == 200
    results = response.json()
    assert len(results) >= 1
    assert results[0]["document_id"] == document.id
    assert results[0]["chunk_id"] >= 1
