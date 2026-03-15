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
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.rag import ask, ask_stream, get_qdrant_client, get_openai_client
from app.cost_tracker import get_summary
from app.guardrails import check_input


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
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class AskRequest(BaseModel):
    question: str
    chat_history: list[ChatMessage] = []

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Is spinal ablation covered under UHC commercial plans?",
                "chat_history": [],
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
    intent: str
    rewritten_query: str | None = None


# ── Endpoints ──────────────────────────────────────────────────────────
@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest):
    """Ask a question about UHC insurance policies."""
    # Input guardrails — reject bad input before hitting the RAG pipeline
    ok, error_msg = check_input(req.question)
    if not ok:
        return JSONResponse(status_code=422, content={"detail": error_msg})

    # Convert chat_history from Pydantic models to dicts for the pipeline
    history = [msg.model_dump() for msg in req.chat_history] if req.chat_history else None

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        partial(
            ask,
            req.question,
            openai_client=openai_client,
            qdrant_client=qdrant_client,
            chat_history=history,
        ),
    )
    return AskResponse(**result)


@app.post("/ask/stream")
async def ask_stream_endpoint(req: AskRequest):
    """Stream a response about UHC insurance policies via SSE."""
    # Input guardrails — same as /ask
    ok, error_msg = check_input(req.question)
    if not ok:
        return JSONResponse(status_code=422, content={"detail": error_msg})

    history = [msg.model_dump() for msg in req.chat_history] if req.chat_history else None

    return StreamingResponse(
        ask_stream(
            req.question,
            openai_client=openai_client,
            qdrant_client=qdrant_client,
            chat_history=history,
        ),
        media_type="text/event-stream",
    )


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
