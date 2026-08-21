from fastapi import APIRouter, HTTPException
from app.schemas import HealthResponse
from app.summarizer import is_ready

router = APIRouter()

@router.get("/healthz", response_model=HealthResponse)
def health_endpoint():
    if not is_ready():
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")   
    return HealthResponse(status="ok", message="Model is loaded and ready.")