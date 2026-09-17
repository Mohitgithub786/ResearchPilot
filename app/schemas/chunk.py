from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    document_id: int
    chunk_text: str
    page_number: int | None
    chunk_order: int
    created_at: datetime
