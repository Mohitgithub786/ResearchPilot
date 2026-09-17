"""
Research agent orchestrator.

Provides the high-level ``run_research()`` entry point that:
1. Builds the initial ResearchState from the user's query
2. Compiles and invokes the LangGraph research workflow
3. Extracts the final report, findings, and sources from the terminal state
4. Returns a structured ResearchResponse
"""

from app.core.logging import get_logger
from app.schemas.research import (
    ResearchResponse,
    ResearchSourceReference,
)
from app.services.research_graph import build_research_graph
from app.services.research_state import ResearchState

logger = get_logger(__name__)


def run_research(query: str) -> ResearchResponse:
    """
    Execute the full agentic research workflow for a given query.

    The graph handles planning, retrieval, analysis, and report writing
    autonomously — including retry loops when retrieval yields no results.
    """
    graph = build_research_graph()

    initial_state: ResearchState = {
        "query": query,
        "plan": [],
        "current_task": "",
        "retrieved_chunks": [],
        "findings": [],
        "report": "",
        "sources": [],
        "retry_count": 0,
        "average_similarity": 0.0,
        "status": "planning",
    }

    logger.info("RESEARCH AGENT: starting research for query: %s", query)
    final_state = graph.invoke(initial_state)
    logger.info(
        "RESEARCH AGENT: completed with status=%s, %d findings, %d sources",
        final_state["status"],
        len(final_state["findings"]),
        len(final_state["sources"]),
    )

    sources = [
        ResearchSourceReference(
            filename=s["filename"],
            page_number=s.get("page_number"),
        )
        for s in final_state["sources"]
    ]

    return ResearchResponse(
        summary=final_state["report"],
        key_findings=final_state["findings"],
        sources=sources,
    )
