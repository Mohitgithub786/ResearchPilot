"""
ChromaDB vector store for chunk embeddings.

Vectors persist under data/chroma_db/. Each Chroma entry uses the SQLite
chunk primary key as its id, maintaining a direct mapping between relational
metadata and vector records.

Collection versioning
---------------------
The collection name is derived automatically from the base name configured in
CHROMA_COLLECTION_NAME and the active embedding model's output dimension:

    {base}_{dim}d    e.g.  researchpilot_chunks_384d

This means switching embedding models (e.g. OpenAI 1536D → MiniLM 384D) will
create a new, correctly-dimensioned collection without touching the old one.
Old collections remain on disk for safety; they can be deleted manually once
you are confident they are no longer needed.
"""

from app.core.logging import get_logger
from typing import Optional

import chromadb

from app.core import config
from app.models.chunk import Chunk
from app.services.embedding_service import get_embedding_dimension, get_embedding_model

logger = get_logger(__name__)

_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def reset_chroma_client() -> None:
    """Clear cached client/collection (used in tests)."""
    global _client, _collection
    _client = None
    _collection = None


def get_chroma_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client


def _resolve_collection_name() -> str:
    """
    Build the dimension-versioned collection name.

    Takes the base name from config (e.g. "researchpilot_chunks") and appends
    the active embedding dimension as a suffix (e.g. "_384d").

    This is computed once per process when get_chroma_collection() is first
    called and then cached via the module-level _collection variable.
    """
    dim = get_embedding_dimension()
    base = config.CHROMA_COLLECTION_NAME
    return f"{base}_{dim}d"


def get_chroma_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = get_chroma_client()
        collection_name = _resolve_collection_name()
        _collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        if _collection.count() > 0:
            logger.info("Using existing collection: %s (%d vectors)", collection_name, _collection.count())
        else:
            logger.info("Created new collection: %s", collection_name)
    return _collection


def _chunk_metadata(chunk: Chunk, source_filename: str) -> dict[str, int | str]:
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "page_number": chunk.page_number if chunk.page_number is not None else -1,
        "source_filename": source_filename,
    }


def index_chunks(chunks: list[Chunk], source_filename: str) -> None:
    """Generate embeddings and upsert chunk vectors into ChromaDB."""
    non_empty = [chunk for chunk in chunks if chunk.chunk_text.strip()]
    if not non_empty:
        return

    embeddings_model = get_embedding_model()
    texts = [chunk.chunk_text for chunk in non_empty]
    vectors = embeddings_model.embed_documents(texts)

    collection = get_chroma_collection()
    collection.add(
        ids=[str(chunk.id) for chunk in non_empty],
        embeddings=vectors,
        documents=texts,
        metadatas=[
            _chunk_metadata(chunk, source_filename) for chunk in non_empty
        ],
    )


def count_indexed_chunks() -> int:
    collection = get_chroma_collection()
    return collection.count()
