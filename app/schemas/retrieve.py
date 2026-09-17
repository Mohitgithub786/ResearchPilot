from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)


class RetrieveResult(BaseModel):
    chunk_id: int
    document_id: int
    page_number: int | None
    source_filename: str
    similarity_score: float
    chunk_text: str


class RetrieveResponse(BaseModel):
    results: list[RetrieveResult]
