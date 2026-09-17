from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    filename: str
    page_number: int | None = None


class ChatRequest(BaseModel):
    session_id: int
    question: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
