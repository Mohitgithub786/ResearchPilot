from fastapi import APIRouter

router = APIRouter(tags=["Integration"])


@router.get("/hello")
def hello(name: str) -> dict[str, str]:
    return {"message": f"Hello {name}, welcome to ResearchPilot."}
