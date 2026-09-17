"""Pydantic schemas for the POST /research endpoint."""

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1)


class ResearchSourceReference(BaseModel):
    filename: str
    page_number: int | None = None


class ResearchResponse(BaseModel):
    summary: str
    key_findings: list[str]
    sources: list[ResearchSourceReference]
