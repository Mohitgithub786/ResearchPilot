from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SessionCreateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: int
    title: str


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int
    role: str
    content: str
    created_at: datetime
