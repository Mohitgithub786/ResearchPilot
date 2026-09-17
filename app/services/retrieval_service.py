"""Similarity search over ChromaDB with SQLite enrichment."""

from sqlalchemy.orm import Session

from app.core.config import RETRIEVAL_TOP_K
from app.models.chunk import Chunk
from app.models.document import Document
from app.schemas.retrieve import RetrieveResult
from app.services.chroma_service import get_chroma_collection
from app.services.embedding_service import get_embedding_model


def _distance_to_similarity(distance: float) -> float:
    """Convert Chroma cosine distance to a similarity score in [0, 1]."""
    return round(max(0.0, 1.0 - distance), 4)


def _page_number_from_metadata(value: int | float | str | None) -> int | None:
    if value is None or value == -1 or value == "-1":
        return None
    return int(value)


def retrieve_chunks(
    db: Session, query: str, top_k: int | None = None
) -> list[RetrieveResult]:
    """
    Embed the query, search ChromaDB, and join results with SQLite metadata.

    Returns an empty list when no vectors are indexed or Chroma finds no matches.
    """
    limit = top_k or RETRIEVAL_TOP_K
    collection = get_chroma_collection()

    if collection.count() == 0:
        return []

    embeddings_model = get_embedding_model()
    query_vector = embeddings_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(limit, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    if not ids:
        return []

    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]

    chunk_ids = [int(chunk_id) for chunk_id in ids]
    sqlite_chunks = {
        chunk.id: chunk
        for chunk in db.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()
    }
    document_ids = {
        chunk.document_id
        for chunk in sqlite_chunks.values()
    }
    documents_by_id = {
        document.id: document
        for document in db.query(Document)
        .filter(Document.id.in_(document_ids))
        .all()
    }

    retrieve_results: list[RetrieveResult] = []
    for index, chunk_id_str in enumerate(ids):
        chunk_id = int(chunk_id_str)
        metadata = metadatas[index] or {}
        sqlite_chunk = sqlite_chunks.get(chunk_id)
        document_id = int(metadata.get("document_id", sqlite_chunk.document_id if sqlite_chunk else 0))
        document = documents_by_id.get(document_id)
        source_filename = str(
            metadata.get("source_filename")
            or (document.filename if document else "unknown")
        )

        retrieve_results.append(
            RetrieveResult(
                chunk_id=chunk_id,
                document_id=document_id,
                page_number=_page_number_from_metadata(metadata.get("page_number")),
                source_filename=source_filename,
                similarity_score=_distance_to_similarity(distances[index]),
                chunk_text=(
                    sqlite_chunk.chunk_text
                    if sqlite_chunk is not None
                    else documents[index]
                ),
            )
        )

    return retrieve_results
