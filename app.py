"""FastAPI interface.  Run:  uvicorn app:app --reload"""
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config
from src.graph import ask, build_rag_graph

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["graph"] = build_rag_graph(index_name=config.PINECONE_INDEX_NAME)
    yield
    _state.clear()


app = FastAPI(title="Agentic AI RAG API", version="1.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, examples=["What is Agentic AI?"])


class Chunk(BaseModel):
    text: str
    page: Optional[int] = None
    score: float


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: List[Chunk]
    confidence_score: float


@app.get("/health")
def health():
    return {"status": "ok", "index": config.PINECONE_INDEX_NAME}


# Plain `def` (not async): the graph makes blocking network calls, so FastAPI
# runs it in a worker thread instead of blocking the event loop.
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be blank.")
    try:
        result = ask(_state["graph"], request.query.strip())
    except Exception as exc:  # surface upstream (OpenAI / Pinecone) failures cleanly
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}") from exc
    return ChatResponse(
        answer=result["answer"],
        retrieved_chunks=result["context"],
        confidence_score=result["score"],
    )
