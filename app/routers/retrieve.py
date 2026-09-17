from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.retrieve import RetrieveRequest, RetrieveResult
from app.services.retrieval_service import retrieve_chunks

router = APIRouter(tags=["Retrieval"])


@router.post("/retrieve", response_model=list[RetrieveResult])
def retrieve(
    payload: RetrieveRequest, db: Session = Depends(get_db)
) -> list[RetrieveResult]:
    return retrieve_chunks(db, payload.query)
