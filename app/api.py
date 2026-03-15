"""
FastAPI backend — thin API layer over the RAG pipeline.

Simple mental model: This is just a web wrapper around app/rag.py.
It receives a question via HTTP, passes it to the RAG pipeline, returns the answer.

Endpoints:
  POST /ask        → Main chat endpoint
  GET  /health     → Health check (for deployment monitoring)
  GET  /costs      → View OpenAI API cost summary
"""

import asyncio
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag import ask, get_qdrant_client, get_openai_client
from app.cost_tracker import get_summary


# ── Lifespan: open/close clients once ─────────────────────────────────
qdrant_client = None
openai_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant_client, openai_client
    qdrant_client = get_qdrant_client()
    openai_client = get_openai_client()
    yield
    if qdrant_client:
        qdrant_client.close()


app = FastAPI(
    title="CombineHealth — Insurance Policy Chatbot",
    description="RAG-powered chatbot for querying UHC insurance policies",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Is spinal ablation covered under UHC commercial plans?"
            }
        }


class Source(BaseModel):
    policy_name: str
    policy_number: str
    source_url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    chunks_used: int


# ── Endpoints ──────────────────────────────────────────────────────────
@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest):
    """Ask a question about UHC insurance policies."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, partial(ask, req.question, openai_client=openai_client, qdrant_client=qdrant_client)
    )
    return AskResponse(**result)


@app.get("/health")
async def health():
    """Health check — returns OK if the service is running and Qdrant is connected."""
    try:
        info = qdrant_client.get_collection("uhc_policies")
        return {
            "status": "healthy",
            "collection": "uhc_policies",
            "indexed_chunks": info.points_count,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/costs")
async def costs():
    """View OpenAI API cost summary."""
    return get_summary()
