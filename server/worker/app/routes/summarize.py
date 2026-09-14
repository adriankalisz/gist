from fastapi import APIRouter, HTTPException
from app.schemas import SummarizeRequest, SummarizeResponse
from app.batcher import enqueue_request

router = APIRouter()

@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_endpoint(request: SummarizeRequest):
    try:
        summary, inference_time = await enqueue_request(request.text)
        return SummarizeResponse(summary=summary, inference_time=inference_time)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))