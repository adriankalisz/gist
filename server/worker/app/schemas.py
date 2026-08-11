from pydantic import BaseModel, Field

class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000) # For research papers, might want to increase max_length to 400000

class SummarizeResponse(BaseModel):
    summary: str
    inference_time: float