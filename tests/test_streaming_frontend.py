"""
Tests for the streaming frontend SSE parser (app/frontend.py — parse_sse_events).

These tests validate the SSE line parser without requiring a Streamlit runtime.
We import the parser function directly and feed it mock Response objects.
"""

import json
from unittest.mock import MagicMock


# ── Import the parser ────────────────────────────────────────────────────
# parse_sse_events lives in app/frontend.py but only depends on requests.Response,
# so we can test it without Streamlit by importing it directly.
# Streamlit's st.* calls happen at module level, so we mock them on import.

import unittest.mock as _mock

# Patch streamlit at import time so module-level st.* calls don't fail
_st_mock = MagicMock()
with _mock.patch.dict("sys.modules", {"streamlit": _st_mock}):
    from app.frontend import parse_sse_events


# ── Helpers ──────────────────────────────────────────────────────────────

def _mock_response(lines: list[str]):
    """Create a mock requests.Response with iter_lines returning the given lines."""
    resp = MagicMock()
    resp.iter_lines.return_value = iter(lines)
    return resp


# ── Basic parsing ────────────────────────────────────────────────────────

def test_parse_single_event():
    resp = _mock_response(['data: {"type": "done"}'])
    events = list(parse_sse_events(resp))
    assert len(events) == 1
    assert events[0] == {"type": "done"}


def test_parse_multiple_events():
    resp = _mock_response([
        'data: {"type": "intent", "intent": "greeting"}',
        'data: {"type": "token", "content": "Hello "}',
        'data: {"type": "token", "content": "there!"}',
        'data: {"type": "sources", "sources": []}',
        'data: {"type": "done"}',
    ])
    events = list(parse_sse_events(resp))
    assert len(events) == 5
    assert events[0]["type"] == "intent"
    assert events[-1]["type"] == "done"


def test_parse_skips_empty_lines():
    """SSE spec uses empty lines as event delimiters — parser should skip them."""
    resp = _mock_response([
        'data: {"type": "intent", "intent": "greeting"}',
        '',
        'data: {"type": "done"}',
        '',
    ])
    events = list(parse_sse_events(resp))
    assert len(events) == 2


def test_parse_skips_comment_lines():
    """SSE lines starting with : are comments — parser should skip them."""
    resp = _mock_response([
        ': this is a comment',
        'data: {"type": "done"}',
    ])
    events = list(parse_sse_events(resp))
    assert len(events) == 1
    assert events[0]["type"] == "done"


def test_parse_skips_non_data_lines():
    """Lines that don't start with 'data: ' should be ignored."""
    resp = _mock_response([
        'event: message',
        'id: 123',
        'data: {"type": "done"}',
        'retry: 5000',
    ])
    events = list(parse_sse_events(resp))
    assert len(events) == 1


def test_parse_skips_malformed_json():
    """Malformed JSON after 'data: ' should be skipped, not crash."""
    resp = _mock_response([
        'data: {"type": "intent", "intent": "greeting"}',
        'data: {broken json',
        'data: {"type": "done"}',
    ])
    events = list(parse_sse_events(resp))
    assert len(events) == 2
    assert events[0]["type"] == "intent"
    assert events[1]["type"] == "done"


# ── Token reconstruction ────────────────────────────────────────────────

def test_tokens_reconstruct_full_answer():
    """Token events should reconstruct the full answer when concatenated."""
    resp = _mock_response([
        'data: {"type": "intent", "intent": "policy_query"}',
        'data: {"type": "token", "content": "Spinal "}',
        'data: {"type": "token", "content": "ablation "}',
        'data: {"type": "token", "content": "is covered."}',
        'data: {"type": "sources", "sources": []}',
        'data: {"type": "done"}',
    ])
    events = list(parse_sse_events(resp))
    token_events = [e for e in events if e["type"] == "token"]
    full = "".join(e["content"] for e in token_events)
    assert full == "Spinal ablation is covered."


# ── Sources parsing ──────────────────────────────────────────────────────

def test_sources_parsed_correctly():
    sources_data = [{"policy_name": "Test", "policy_number": "TP-001", "source_url": "https://example.com"}]
    resp = _mock_response([
        f'data: {json.dumps({"type": "sources", "sources": sources_data})}',
        'data: {"type": "done"}',
    ])
    events = list(parse_sse_events(resp))
    sources_event = [e for e in events if e["type"] == "sources"][0]
    assert len(sources_event["sources"]) == 1
    assert sources_event["sources"][0]["policy_number"] == "TP-001"