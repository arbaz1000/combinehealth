"""
FastAPI backend — thin API layer over the RAG pipeline.

Simple mental model: This is just a web wrapper around app/rag.py.
It receives a question via HTTP, passes it to the RAG pipeline, returns the answer.

Endpoints:
  POST /ask        → Main chat endpoint
  GET  /health     → Health check (for deployment monitoring)
  GET  /costs      → View OpenAI API cost summary
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.rag import ask, ask_stream, get_qdrant_client, get_openai_client
from app.cost_tracker import get_summary
from app.guardrails import check_input
from app.config import INSURER_SHORT_NAME, QDRANT_COLLECTION


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
    description=f"RAG-powered chatbot for querying {INSURER_SHORT_NAME} insurance policies",
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
                "question": f"Is spinal ablation covered under {INSURER_SHORT_NAME} commercial plans?",
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
    """Ask a question about insurance policies."""
    # Input guardrails — reject empty/too-long, redact PII
    ok, sanitized, _redacted = check_input(req.question)
    if not ok:
        return JSONResponse(status_code=422, content={"detail": sanitized})

    # Convert chat_history from Pydantic models to dicts for the pipeline
    history = [msg.model_dump() for msg in req.chat_history] if req.chat_history else None

    result = await ask(
        sanitized,
        openai_client=openai_client,
        qdrant_client=qdrant_client,
        chat_history=history,
    )
    return AskResponse(**result)


@app.post("/ask/stream")
async def ask_stream_endpoint(req: AskRequest):
    """Stream a response about insurance policies via SSE."""
    # Input guardrails — reject empty/too-long, redact PII
    ok, sanitized, _redacted = check_input(req.question)
    if not ok:
        return JSONResponse(status_code=422, content={"detail": sanitized})

    history = [msg.model_dump() for msg in req.chat_history] if req.chat_history else None

    return StreamingResponse(
        ask_stream(
            sanitized,
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
        info = qdrant_client.get_collection(QDRANT_COLLECTION)
        return {
            "status": "healthy",
            "collection": QDRANT_COLLECTION,
            "indexed_chunks": info.points_count,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/costs")
async def costs():
    """View OpenAI API cost summary."""
    return get_summary()
