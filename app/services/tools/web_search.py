from app.core.logging import get_logger

logger = get_logger(__name__)

def web_search(query: str, num_results: int = 5) -> list[dict]:
    """Perform a web search using DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
            logger.info("WEB SEARCH: found %d results for query: %s", len(results), query)
            return results
    except Exception as e:
        logger.error("WEB SEARCH: error performing search: %s", e)
        return []
