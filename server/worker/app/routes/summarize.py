from fastapi import APIRouter, HTTPException
from app.schemas import SummarizeRequest, SummarizeResponse
from app.summarizer import summarize

router = APIRouter()

@router.post("/summarize", response_model=SummarizeResponse)
def summarize_endpoint(request: SummarizeRequest):
    try:
        summary, inference_time = summarize(request.text)
        return SummarizeResponse(summary=summary, inference_time=inference_time)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))