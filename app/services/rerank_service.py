from typing import List
from app.schemas.retrieve import RetrieveResult

def rrf_score(dense_rank: int, sparse_rank: int, k=60) -> float:
    score = 0.0
    if dense_rank is not None:
        score += 1.0 / (k + dense_rank)
    if sparse_rank is not None:
        score += 1.0 / (k + sparse_rank)
    return score

def get_reranker():
    try:
        from flashrank import Ranker, RerankRequest
        return Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
    except ImportError:
        return None

def rerank_results(query: str, results: List[RetrieveResult], top_k: int = 5) -> List[RetrieveResult]:
    ranker = get_reranker()
    if not ranker or not results:
        return sorted(results, key=lambda x: x.similarity_score, reverse=True)[:top_k]
    
    from flashrank import RerankRequest
    passages = [
        {
            "id": res.chunk_id,
            "text": res.chunk_text,
            "meta": {"res": res}
        }
        for res in results
    ]
    
    rerankrequest = RerankRequest(query=query, passages=passages)
    reranked_results = ranker.rerank(rerankrequest)
    
    final_results = []
    for item in reranked_results:
        res = item['meta']['res']
        # Update score to reranker score
        res.similarity_score = item['score']
        final_results.append(res)
        
    return final_results[:top_k]
