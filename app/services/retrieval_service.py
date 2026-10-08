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


from app.services.rerank_service import rerank_results, rrf_score

def retrieve_chunks(
    db: Session, query: str, top_k: int | None = None
) -> list[RetrieveResult]:
    limit = top_k or RETRIEVAL_TOP_K
    collection = get_chroma_collection()

    if collection.count() == 0:
        return []

    # 1. Dense retrieval
    embeddings_model = get_embedding_model()
    query_vector = embeddings_model.embed_query(query)
    
    dense_results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(20, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    
    dense_ids = dense_results.get("ids", [[]])[0]
    dense_chunk_ids = [int(cid) for cid in dense_ids]
    
    # 2. Sparse retrieval (BM25)
    all_chunks = db.query(Chunk).all()
    sparse_chunk_ids = []
    if all_chunks:
        try:
            from rank_bm25 import BM25Okapi
            corpus = [c.chunk_text.split() for c in all_chunks]
            bm25 = BM25Okapi(corpus)
            tokenized_query = query.split()
            scores = bm25.get_scores(tokenized_query)
            # get top 20
            top_sparse_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:20]
            sparse_chunk_ids = [all_chunks[i].id for i in top_sparse_idx if scores[i] > 0]
        except ImportError:
            pass

    # Combine using RRF
    combined_ids = set(dense_chunk_ids + sparse_chunk_ids)
    
    sqlite_chunks = {
        chunk.id: chunk
        for chunk in db.query(Chunk).filter(Chunk.id.in_(combined_ids)).all()
    }
    document_ids = {chunk.document_id for chunk in sqlite_chunks.values()}
    documents_by_id = {
        document.id: document
        for document in db.query(Document).filter(Document.id.in_(document_ids)).all()
    }
    
    retrieve_results: list[RetrieveResult] = []
    
    for chunk_id in combined_ids:
        chunk = sqlite_chunks.get(chunk_id)
        if not chunk: continue
        
        dense_rank = dense_chunk_ids.index(chunk_id) + 1 if chunk_id in dense_chunk_ids else None
        sparse_rank = sparse_chunk_ids.index(chunk_id) + 1 if chunk_id in sparse_chunk_ids else None
        
        score = rrf_score(dense_rank, sparse_rank)
        
        document = documents_by_id.get(chunk.document_id)
        source_filename = document.filename if document else "unknown"
        
        retrieve_results.append(
            RetrieveResult(
                chunk_id=chunk_id,
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                source_filename=source_filename,
                similarity_score=score,
                chunk_text=chunk.chunk_text,
            )
        )
        
    # Sort by RRF score before reranking
    retrieve_results = sorted(retrieve_results, key=lambda x: x.similarity_score, reverse=True)
    
    # 3. Rerank
    final_results = rerank_results(query, retrieve_results, top_k=limit)
    
    return final_results
