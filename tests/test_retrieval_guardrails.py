"""
Tests for retrieval guardrails (app/guardrails.py — check_retrieval).

Tests use synthetic chunk dicts with controlled scores to verify filtering
and confidence tier assignment without hitting Qdrant or OpenAI.
"""

from app.guardrails import check_retrieval


# ── Helpers ──────────────────────────────────────────────────────────────

def _chunk(score: float, text: str = "sample text") -> dict:
    """Create a minimal chunk dict with a given score."""
    return {
        "text": text,
        "policy_name": "Test Policy",
        "policy_number": "TP-001",
        "section_name": "Coverage",
        "source_url": "https://example.com",
        "score": score,
    }


# ── High confidence ─────────────────────────────────────────────────────

def test_high_confidence_all_above_threshold():
    chunks = [_chunk(0.85), _chunk(0.72), _chunk(0.55)]
    filtered, confidence = check_retrieval(chunks)
    assert confidence == "high"
    assert len(filtered) == 3


def test_high_confidence_some_filtered():
    """Chunks below 0.35 are dropped but best is still >= 0.5."""
    chunks = [_chunk(0.75), _chunk(0.50), _chunk(0.30), _chunk(0.20)]
    filtered, confidence = check_retrieval(chunks)
    assert confidence == "high"
    assert len(filtered) == 2
    assert all(c["score"] >= 0.35 for c in filtered)


# ── Low confidence ──────────────────────────────────────────────────────

def test_low_confidence_best_below_half():
    """All surviving chunks score between 0.35 and 0.5 → low confidence."""
    chunks = [_chunk(0.48), _chunk(0.40), _chunk(0.36)]
    filtered, confidence = check_retrieval(chunks)
    assert confidence == "low"
    assert len(filtered) == 3


def test_low_confidence_with_some_dropped():
    chunks = [_chunk(0.45), _chunk(0.30), _chunk(0.10)]
    filtered, confidence = check_retrieval(chunks)
    assert confidence == "low"
    assert len(filtered) == 1
    assert filtered[0]["score"] == 0.45


# ── No confidence (none) ────────────────────────────────────────────────

def test_none_confidence_all_below_threshold():
    chunks = [_chunk(0.30), _chunk(0.20), _chunk(0.10)]
    filtered, confidence = check_retrieval(chunks)
    assert confidence == "none"
    assert len(filtered) == 0


def test_none_confidence_empty_input():
    filtered, confidence = check_retrieval([])
    assert confidence == "none"
    assert len(filtered) == 0


# ── Edge cases ──────────────────────────────────────────────────────────

def test_exact_threshold_included():
    """Chunk scoring exactly 0.35 should survive."""
    chunks = [_chunk(0.35)]
    filtered, confidence = check_retrieval(chunks)
    assert len(filtered) == 1


def test_exact_low_confidence_boundary():
    """Chunk scoring exactly 0.5 should be high confidence."""
    chunks = [_chunk(0.50)]
    filtered, confidence = check_retrieval(chunks)
    assert confidence == "high"


def test_chunk_missing_score_key_treated_as_zero():
    """Chunk without a score key defaults to 0 and gets filtered out."""
    chunks = [{"text": "no score", "policy_name": "X", "policy_number": "X"}]
    filtered, confidence = check_retrieval(chunks)
    assert confidence == "none"
    assert len(filtered) == 0


def test_ordering_preserved():
    """Filtered chunks maintain their original order."""
    chunks = [_chunk(0.80), _chunk(0.20), _chunk(0.60)]
    filtered, _ = check_retrieval(chunks)
    assert [c["score"] for c in filtered] == [0.80, 0.60]


def test_single_high_score_chunk():
    chunks = [_chunk(0.92)]
    filtered, confidence = check_retrieval(chunks)
    assert confidence == "high"
    assert len(filtered) == 1