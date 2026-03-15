"""
Tests for SSE streaming (app/rag.py — ask_stream, app/api.py — /ask/stream).

Tests use mocked OpenAI and Qdrant clients to verify SSE event structure
and sequencing without hitting external services.
"""

import json
from unittest.mock import MagicMock, patch

from app.rag import ask_stream, _sse_event


# ── Helpers ──────────────────────────────────────────────────────────────

def _parse_events(generator) -> list[dict]:
    """Collect SSE events from a generator, parse each into a dict."""
    events = []
    for raw in generator:
        assert raw.startswith("data: "), f"SSE line must start with 'data: ', got: {raw!r}"
        assert raw.endswith("\n\n"), f"SSE line must end with double newline, got: {raw!r}"
        payload = raw[len("data: "):-2]  # strip "data: " prefix and "\n\n" suffix
        events.append(json.loads(payload))
    return events


def _chunk(score: float, text: str = "Sample policy text.") -> dict:
    """Create a minimal chunk dict with a given score."""
    return {
        "text": text,
        "policy_name": "Test Policy",
        "policy_number": "TP-001",
        "section_name": "Coverage",
        "source_url": "https://example.com/tp001",
        "score": score,
    }


def _mock_stream_chunks(tokens: list[str]):
    """
    Create mock OpenAI streaming chunks.

    Each token becomes a chunk with delta.content set.
    Final chunk has usage stats and no content.
    """
    chunks = []
    for token in tokens:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = token
        chunk.usage = None
        chunks.append(chunk)

    # Final chunk with usage (no content)
    final = MagicMock()
    final.choices = []
    final.usage = MagicMock()
    final.usage.prompt_tokens = 100
    final.usage.completion_tokens = len(tokens)
    chunks.append(final)

    return chunks


# ── SSE event formatting ────────────────────────────────────────────────

def test_sse_event_format():
    """_sse_event produces correct SSE data line."""
    result = _sse_event({"type": "done"})
    assert result == 'data: {"type": "done"}\n\n'


def test_sse_event_with_content():
    result = _sse_event({"type": "token", "content": "hello"})
    parsed = json.loads(result[len("data: "):-2])
    assert parsed == {"type": "token", "content": "hello"}


# ── Greeting intent (Tier 1 regex — no OpenAI call) ────────────────────

def test_stream_greeting_word_by_word():
    """Greeting should stream canned response word-by-word via SSE."""
    events = _parse_events(ask_stream(
        "Hi",
        openai_client=MagicMock(),
        qdrant_client=MagicMock(),
    ))

    # First event: intent
    assert events[0]["type"] == "intent"
    assert events[0]["intent"] == "greeting"

    # Middle events: tokens (word-by-word)
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) > 1, "Canned response should be split into multiple tokens"

    # Reconstruct the full response
    full_response = "".join(e["content"] for e in token_events)
    assert "UHC policy assistant" in full_response

    # Sources event (empty for greetings)
    sources_events = [e for e in events if e["type"] == "sources"]
    assert len(sources_events) == 1
    assert sources_events[0]["sources"] == []

    # Done event
    assert events[-1]["type"] == "done"


def test_stream_thanks():
    """Thanks (Tier 1) also streams word-by-word."""
    events = _parse_events(ask_stream(
        "Thanks!",
        openai_client=MagicMock(),
        qdrant_client=MagicMock(),
    ))

    assert events[0]["type"] == "intent"
    assert events[0]["intent"] == "greeting"

    token_events = [e for e in events if e["type"] == "token"]
    full = "".join(e["content"] for e in token_events)
    assert "welcome" in full.lower()
    assert events[-1]["type"] == "done"


# ── Event sequence order ────────────────────────────────────────────────

def test_stream_event_order_greeting():
    """Events must arrive in order: intent → token(s) → sources → done."""
    events = _parse_events(ask_stream(
        "Hello!",
        openai_client=MagicMock(),
        qdrant_client=MagicMock(),
    ))

    types = [e["type"] for e in events]
    # Should start with intent, end with done, sources just before done
    assert types[0] == "intent"
    assert types[-1] == "done"
    assert types[-2] == "sources"
    # Everything in between should be tokens
    for t in types[1:-2]:
        assert t == "token"


@patch("app.rag.classify_intent")
@patch("app.rag.retrieve")
@patch("app.rag.generate_answer_stream")
def test_stream_event_order_policy_query(mock_gen_stream, mock_retrieve, mock_classify):
    """Policy query: intent → tokens → sources → done."""
    mock_classify.return_value = {
        "intent": "policy_query",
        "response": None,
        "rewritten_query": "Is spinal ablation covered?",
    }
    mock_retrieve.return_value = [_chunk(0.85, "Ablation is covered.")]
    mock_gen_stream.return_value = iter(["Spinal ", "ablation ", "is ", "covered."])

    events = _parse_events(ask_stream(
        "Is spinal ablation covered?",
        openai_client=MagicMock(),
        qdrant_client=MagicMock(),
    ))

    types = [e["type"] for e in events]
    assert types[0] == "intent"
    assert types[-1] == "done"
    assert types[-2] == "sources"
    for t in types[1:-2]:
        assert t == "token"


# ── Policy query streaming ──────────────────────────────────────────────

@patch("app.rag.classify_intent")
@patch("app.rag.retrieve")
@patch("app.rag.generate_answer_stream")
def test_stream_policy_query_tokens(mock_gen_stream, mock_retrieve, mock_classify):
    """Policy query tokens arrive individually and reconstruct the answer."""
    mock_classify.return_value = {
        "intent": "policy_query",
        "response": None,
        "rewritten_query": "Is ablation covered?",
    }
    mock_retrieve.return_value = [_chunk(0.85, "Ablation is covered under policy TP-001.")]
    mock_gen_stream.return_value = iter(["Yes, ", "ablation ", "is ", "covered."])

    events = _parse_events(ask_stream(
        "Is ablation covered?",
        openai_client=MagicMock(),
        qdrant_client=MagicMock(),
    ))

    # Check intent
    assert events[0]["intent"] == "policy_query"
    assert events[0]["rewritten_query"] == "Is ablation covered?"

    # Check tokens
    token_events = [e for e in events if e["type"] == "token"]
    full = "".join(e["content"] for e in token_events)
    assert full == "Yes, ablation is covered."

    # Check sources
    sources_event = [e for e in events if e["type"] == "sources"][0]
    assert len(sources_event["sources"]) == 1
    assert sources_event["sources"][0]["policy_number"] == "TP-001"


@patch("app.rag.classify_intent")
@patch("app.rag.retrieve")
@patch("app.rag.generate_answer_stream")
def test_stream_policy_query_with_rewritten_query(mock_gen_stream, mock_retrieve, mock_classify):
    """Follow-up intent sends rewritten_query in the intent event."""
    mock_classify.return_value = {
        "intent": "follow_up",
        "response": None,
        "rewritten_query": "What CPT codes apply to spinal ablation under UHC?",
    }
    mock_retrieve.return_value = [_chunk(0.70)]
    mock_gen_stream.return_value = iter(["CPT ", "codes."])

    events = _parse_events(ask_stream(
        "What CPT codes apply?",
        openai_client=MagicMock(),
        qdrant_client=MagicMock(),
        chat_history=[
            {"role": "user", "content": "Is spinal ablation covered?"},
            {"role": "assistant", "content": "Yes it is."},
        ],
    ))

    assert events[0]["type"] == "intent"
    assert events[0]["intent"] == "follow_up"
    assert events[0]["rewritten_query"] == "What CPT codes apply to spinal ablation under UHC?"


# ── No retrieval results ────────────────────────────────────────────────

@patch("app.rag.classify_intent")
@patch("app.rag.retrieve")
@patch("app.rag.generate_answer_stream")
def test_stream_no_retrieval_results(mock_gen_stream, mock_retrieve, mock_classify):
    """When no chunks survive filtering, LLM is still called (closed-book)."""
    mock_classify.return_value = {
        "intent": "policy_query",
        "response": None,
        "rewritten_query": "Does UHC cover time travel?",
    }
    mock_retrieve.return_value = [_chunk(0.10), _chunk(0.20)]  # all below threshold
    mock_gen_stream.return_value = iter(["No ", "matching ", "policy ", "found."])

    events = _parse_events(ask_stream(
        "Does UHC cover time travel?",
        openai_client=MagicMock(),
        qdrant_client=MagicMock(),
    ))

    token_events = [e for e in events if e["type"] == "token"]
    full = "".join(e["content"] for e in token_events)
    assert "No matching policy found." == full

    # Sources should be empty
    sources_event = [e for e in events if e["type"] == "sources"][0]
    assert sources_event["sources"] == []

    # generate_answer_stream should have been called with confidence="none"
    call_kwargs = mock_gen_stream.call_args
    assert call_kwargs[1]["retrieval_confidence"] == "none" or call_kwargs[0][1] == ""


# ── Off-topic streaming ─────────────────────────────────────────────────

@patch("app.rag.classify_intent")
def test_stream_off_topic(mock_classify):
    """Off-topic intent streams canned response word-by-word, no retrieval."""
    mock_classify.return_value = {
        "intent": "off_topic",
        "response": "I can only help with UHC insurance policy questions.",
        "rewritten_query": None,
    }

    events = _parse_events(ask_stream(
        "What is the weather?",
        openai_client=MagicMock(),
        qdrant_client=MagicMock(),
    ))

    assert events[0]["intent"] == "off_topic"

    token_events = [e for e in events if e["type"] == "token"]
    full = "".join(e["content"] for e in token_events)
    assert full == "I can only help with UHC insurance policy questions."
    assert events[-1]["type"] == "done"


# ── API endpoint ─────────────────────────────────────────────────────────

def test_stream_endpoint_input_guardrail():
    """The /ask/stream endpoint should reject bad input with 422."""
    from fastapi.testclient import TestClient
    from app.api import app

    client = TestClient(app)

    # Empty question
    resp = client.post("/ask/stream", json={"question": ""})
    assert resp.status_code == 422

    # PII (SSN)
    resp = client.post("/ask/stream", json={"question": "My SSN is 123-45-6789"})
    assert resp.status_code == 422


def test_stream_endpoint_returns_event_stream():
    """The /ask/stream endpoint returns text/event-stream content type for valid requests."""
    from unittest.mock import patch as _patch
    from fastapi.testclient import TestClient
    from app.api import app
    import app.api as api_module

    # Mock the global clients so lifespan doesn't try to open Qdrant
    api_module.qdrant_client = MagicMock()
    api_module.openai_client = MagicMock()

    try:
        with _patch("app.api.ask_stream") as mock_ask_stream:
            # Simulate a greeting SSE stream
            mock_ask_stream.return_value = iter([
                _sse_event({"type": "intent", "intent": "greeting", "rewritten_query": None}),
                _sse_event({"type": "token", "content": "Hello! "}),
                _sse_event({"type": "token", "content": "How can I help?"}),
                _sse_event({"type": "sources", "sources": []}),
                _sse_event({"type": "done"}),
            ])

            client = TestClient(app, raise_server_exceptions=True)
            resp = client.post("/ask/stream", json={"question": "Hello!"})
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

            # Parse events from response body
            events = []
            for line in resp.text.strip().split("\n\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line[len("data: "):]))

            assert events[0]["type"] == "intent"
            assert events[-1]["type"] == "done"
    finally:
        api_module.qdrant_client = None
        api_module.openai_client = None