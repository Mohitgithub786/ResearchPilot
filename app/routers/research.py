"""POST /research endpoint — agentic research workflow."""

from fastapi import APIRouter

from app.schemas.research import ResearchRequest, ResearchResponse
from app.services.research_agent import run_research

router = APIRouter(tags=["Research"])


@router.post("/research", response_model=ResearchResponse)
def research(payload: ResearchRequest) -> ResearchResponse:
    """Run the agentic research pipeline and return a structured report."""
    return run_research(payload.query)

@router.get("/stream")
async def stream_research_endpoint(query: str):
    from fastapi.responses import StreamingResponse
    from app.services.research_agent import stream_research
    return StreamingResponse(stream_research(query), media_type="text/event-stream")

@router.post("/export")
async def export_research(payload: ResearchRequest, format: str = "pdf"):
    from fastapi.responses import StreamingResponse, PlainTextResponse
    from app.services.research_agent import run_research
    from app.services.exporter_service import to_pdf, to_markdown
    
    # Run the research synchronously to get the data
    research_data = run_research(payload.query)
    data_dict = {
        "summary": research_data.summary,
        "key_findings": research_data.key_findings,
        "sources": [{"filename": s.filename, "page_number": s.page_number} for s in research_data.sources]
    }
    
    if format == "pdf":
        pdf_bytes = to_pdf(data_dict)
        return StreamingResponse(
            pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=report.pdf"}
        )
    else:
        md = to_markdown(data_dict)
        return PlainTextResponse(
            md,
            media_type="text/markdown",
            headers={"Content-Disposition": "attachment; filename=report.md"}
        )


