from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routes import summarize
from app.summarizer import load_model
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="gist-worker", lifespan=lifespan)
app.include_router(summarize.router)

if __name__ == "__main__":
    uvicorn.run(f"app.main:app", host="0.0.0.0", port=8000, reload=True)
