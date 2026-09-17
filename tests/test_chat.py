
from app.services.llm_service import TestLLMService, get_llm_service
from app.services.prompt_builder import build_rag_prompt
from app.services.rag_service import NO_CONTEXT_ANSWER, build_citations
from app.schemas.retrieve import RetrieveResult

from tests.helpers import seed_document_with_chunks


def _create_session(test_client) -> int:
    """Helper to create a session and return its id."""
    resp = test_client.post("/sessions")
    assert resp.status_code == 201
    return resp.json()["session_id"]


def test_successful_grounded_answer_generation(client):
    test_client, session_factory, _chroma_dir = client
    session_id = _create_session(test_client)
    seed_document_with_chunks(
        session_factory,
        "attention.pdf",
        [
            "Positional encoding injects order information into transformer inputs.",
        ],
    )

    response = test_client.post(
        "/chat",
        json={"session_id": session_id, "question": "What is positional encoding?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "positional encoding" in body["answer"].lower()
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["filename"] == "attention.pdf"


def test_citation_generation_and_deduplication(client):
    test_client, session_factory, _chroma_dir = client
    session_id = _create_session(test_client)
    seed_document_with_chunks(
        session_factory,
        "attention.pdf",
        [
            "Positional encoding injects order information into transformer inputs.",
            "Positional encoding is added before the self-attention layers.",
        ],
        page_numbers=[1, 1],
    )

    response = test_client.post(
        "/chat",
        json={"session_id": session_id, "question": "Explain positional encoding."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == [{"filename": "attention.pdf", "page_number": 1}]


def test_citation_multiple_pages(client):
    chunks = [
        RetrieveResult(
            chunk_id=1,
            document_id=1,
            page_number=3,
            source_filename="attention.pdf",
            similarity_score=0.9,
            chunk_text="Chunk on page 3.",
        ),
        RetrieveResult(
            chunk_id=2,
            document_id=1,
            page_number=7,
            source_filename="attention.pdf",
            similarity_score=0.8,
            chunk_text="Chunk on page 7.",
        ),
    ]

    sources = build_citations(chunks)
    assert len(sources) == 2
    assert sources[0].page_number == 3
    assert sources[1].page_number == 7


def test_empty_retrieval_handling(client):
    test_client, _session_factory, _chroma_dir = client
    session_id = _create_session(test_client)

    response = test_client.post(
        "/chat",
        json={"session_id": session_id, "question": "What is positional encoding?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == NO_CONTEXT_ANSWER
    assert body["sources"] == []


def test_multiple_document_retrieval(client):
    test_client, session_factory, _chroma_dir = client
    session_id = _create_session(test_client)
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
        "/chat",
        json={"session_id": session_id, "question": "What is positional encoding?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sources"][0]["filename"] == "attention.pdf"
    assert "positional encoding" in body["answer"].lower()


def test_llm_provider_abstraction(monkeypatch):
    get_llm_service.cache_clear()
    monkeypatch.setenv("LLM_PROVIDER", "test")

    service = get_llm_service()

    assert isinstance(service, TestLLMService)
    prompt = build_rag_prompt(
        "What is positional encoding?",
        [
            RetrieveResult(
                chunk_id=1,
                document_id=1,
                page_number=1,
                source_filename="attention.pdf",
                similarity_score=0.95,
                chunk_text=(
                    "Positional encoding injects order information into "
                    "transformer inputs."
                ),
            )
        ],
    )
    answer = service.generate(prompt)
    assert "positional encoding" in answer.lower()


def test_prompt_builder_grounding_instruction():
    prompt = build_rag_prompt(
        "What is attention?",
        [
            RetrieveResult(
                chunk_id=1,
                document_id=1,
                page_number=2,
                source_filename="attention.pdf",
                similarity_score=0.9,
                chunk_text="Self-attention computes weighted relationships.",
            )
        ],
    )

    assert "Answer ONLY using the provided context" in prompt
    assert "attention.pdf, page 2" in prompt
    assert "What is attention?" in prompt
