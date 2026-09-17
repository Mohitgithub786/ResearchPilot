"""POST /research endpoint — agentic research workflow."""

from fastapi import APIRouter

from app.schemas.research import ResearchRequest, ResearchResponse
from app.services.research_agent import run_research

router = APIRouter(tags=["Research"])


@router.post("/research", response_model=ResearchResponse)
def research(payload: ResearchRequest) -> ResearchResponse:
    """Run the agentic research pipeline and return a structured report."""
    return run_research(payload.query)
