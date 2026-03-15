"""
Tests for intent classification + query rewriting (app/classifier.py).

Tier 1 tests: regex-based, no mocking needed.
Tier 2 tests: mock the OpenAI client to avoid real API calls.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.classifier import classify_intent


# ── Tier 1: Regex greetings (no API call) ───────────────────────────────

@pytest.mark.asyncio
async def test_greeting_hi():
    result = await classify_intent("Hi")
    assert result["intent"] == "greeting"
    assert result["response"] is not None
    assert result["rewritten_query"] is None


@pytest.mark.asyncio
async def test_greeting_hello():
    result = await classify_intent("hello!")
    assert result["intent"] == "greeting"


@pytest.mark.asyncio
async def test_greeting_hey():
    result = await classify_intent("Hey")
    assert result["intent"] == "greeting"


@pytest.mark.asyncio
async def test_greeting_good_morning():
    result = await classify_intent("Good morning")
    assert result["intent"] == "greeting"


@pytest.mark.asyncio
async def test_greeting_howdy():
    result = await classify_intent("Howdy!")
    assert result["intent"] == "greeting"


@pytest.mark.asyncio
async def test_greeting_whats_up():
    result = await classify_intent("What's up")
    assert result["intent"] == "greeting"


# ── Tier 1: Thanks (no API call) ───────────────────────────────────────

@pytest.mark.asyncio
async def test_thanks():
    result = await classify_intent("Thanks!")
    assert result["intent"] == "greeting"
    assert "welcome" in result["response"].lower()


@pytest.mark.asyncio
async def test_thank_you():
    result = await classify_intent("Thank you")
    assert result["intent"] == "greeting"


@pytest.mark.asyncio
async def test_great_thanks():
    result = await classify_intent("Great, thanks!")
    assert result["intent"] == "greeting"


@pytest.mark.asyncio
async def test_got_it():
    result = await classify_intent("Got it")
    assert result["intent"] == "greeting"


# ── Tier 1: Bye (no API call) ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_bye():
    result = await classify_intent("Bye")
    assert result["intent"] == "greeting"
    assert "goodbye" in result["response"].lower() or "come back" in result["response"].lower()


@pytest.mark.asyncio
async def test_goodbye():
    result = await classify_intent("Goodbye!")
    assert result["intent"] == "greeting"


@pytest.mark.asyncio
async def test_take_care():
    result = await classify_intent("Take care")
    assert result["intent"] == "greeting"


# ── Tier 1: These should NOT match regex (fall through to Tier 2) ──────

@pytest.mark.asyncio
async def test_hi_with_question_falls_through():
    """'Hi, is ablation covered?' is NOT a simple greeting — needs Tier 2."""
    # This won't match the regex because there's text after the greeting
    # We need to mock Tier 2 for this
    mock_client = _make_mock_client({
        "intent": "policy_query",
        "response": None,
        "rewritten_query": "Hi, is ablation covered?",
    })
    result = await classify_intent("Hi, is ablation covered?", openai_client=mock_client)
    assert result["intent"] == "policy_query"


# ── Tier 2: Off-topic (mocked LLM) ────────────────────────────────────

@pytest.mark.asyncio
async def test_off_topic_weather():
    mock_client = _make_mock_client({
        "intent": "off_topic",
        "response": "I'm designed to help with UHC insurance policy questions. I can't help with weather, but feel free to ask about coverage or CPT codes!",
        "rewritten_query": None,
    })
    result = await classify_intent("What's the weather today?", openai_client=mock_client)
    assert result["intent"] == "off_topic"
    assert result["response"] is not None
    assert result["rewritten_query"] is None


@pytest.mark.asyncio
async def test_off_topic_sports():
    mock_client = _make_mock_client({
        "intent": "off_topic",
        "response": "I specialize in UHC insurance policies. For sports news, you'd want a different resource. Can I help with any policy questions?",
        "rewritten_query": None,
    })
    result = await classify_intent("Who won the Super Bowl?", openai_client=mock_client)
    assert result["intent"] == "off_topic"


@pytest.mark.asyncio
async def test_off_topic_coding():
    mock_client = _make_mock_client({
        "intent": "off_topic",
        "response": "That's a programming question — not my area! I'm here to help with UHC coverage and policy questions.",
        "rewritten_query": None,
    })
    result = await classify_intent("How do I write a for loop in Python?", openai_client=mock_client)
    assert result["intent"] == "off_topic"


# ── Tier 2: Policy query (mocked LLM) ─────────────────────────────────

@pytest.mark.asyncio
async def test_policy_query_direct():
    mock_client = _make_mock_client({
        "intent": "policy_query",
        "response": None,
        "rewritten_query": "Is spinal ablation covered under UHC commercial plans?",
    })
    result = await classify_intent(
        "Is spinal ablation covered under UHC commercial plans?",
        openai_client=mock_client,
    )
    assert result["intent"] == "policy_query"
    assert result["rewritten_query"] is not None
    assert result["response"] is None


@pytest.mark.asyncio
async def test_policy_query_cpt():
    mock_client = _make_mock_client({
        "intent": "policy_query",
        "response": None,
        "rewritten_query": "What CPT codes apply to cardiac catheterization?",
    })
    result = await classify_intent(
        "What CPT codes apply to cardiac catheterization?",
        openai_client=mock_client,
    )
    assert result["intent"] == "policy_query"


@pytest.mark.asyncio
async def test_policy_query_prior_auth():
    mock_client = _make_mock_client({
        "intent": "policy_query",
        "response": None,
        "rewritten_query": "Does knee arthroscopy require prior authorization?",
    })
    result = await classify_intent(
        "Does knee arthroscopy require prior authorization?",
        openai_client=mock_client,
    )
    assert result["intent"] == "policy_query"


# ── Tier 2: Follow-up with rewriting (mocked LLM) ─────────────────────

@pytest.mark.asyncio
async def test_follow_up_rewrite():
    """Follow-up referencing prior context should be rewritten."""
    history = [
        {"role": "user", "content": "Is spinal ablation covered under UHC?"},
        {"role": "assistant", "content": "Yes, spinal ablation is covered under policy 2024T0538..."},
    ]
    mock_client = _make_mock_client({
        "intent": "follow_up",
        "response": None,
        "rewritten_query": "What CPT codes apply to spinal ablation under UHC?",
    })
    result = await classify_intent(
        "What CPT codes apply?",
        chat_history=history,
        openai_client=mock_client,
    )
    assert result["intent"] == "follow_up"
    assert "spinal ablation" in result["rewritten_query"].lower()
    assert result["response"] is None


@pytest.mark.asyncio
async def test_follow_up_pronoun_resolution():
    """Pronouns like 'it' and 'that' should be resolved using history."""
    history = [
        {"role": "user", "content": "Tell me about lumbar spinal fusion coverage."},
        {"role": "assistant", "content": "Lumbar spinal fusion is covered under..."},
    ]
    mock_client = _make_mock_client({
        "intent": "follow_up",
        "response": None,
        "rewritten_query": "Does lumbar spinal fusion require prior authorization under UHC?",
    })
    result = await classify_intent(
        "Does it require prior auth?",
        chat_history=history,
        openai_client=mock_client,
    )
    assert result["intent"] == "follow_up"
    assert "lumbar" in result["rewritten_query"].lower()


# ── Edge cases ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_malformed_llm_json_fallback():
    """If LLM returns invalid JSON, default to policy_query (don't block user)."""
    mock_client = _make_mock_client_raw("This is not valid JSON at all")
    result = await classify_intent(
        "Is ablation covered?",
        openai_client=mock_client,
    )
    assert result["intent"] == "policy_query"


@pytest.mark.asyncio
async def test_empty_history_treated_as_no_history():
    """Empty chat_history should work the same as None."""
    mock_client = _make_mock_client({
        "intent": "policy_query",
        "response": None,
        "rewritten_query": "Is ablation covered?",
    })
    result = await classify_intent("Is ablation covered?", chat_history=[], openai_client=mock_client)
    assert result["intent"] == "policy_query"


@pytest.mark.asyncio
async def test_cost_tracking_called():
    """Tier 2 calls should be cost-tracked."""
    mock_client = _make_mock_client({
        "intent": "policy_query",
        "response": None,
        "rewritten_query": "Is ablation covered?",
    })
    with patch("app.classifier.log_call") as mock_log:
        await classify_intent("Is ablation covered?", openai_client=mock_client)
        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args
        assert call_kwargs[1]["metadata"]["purpose"] == "intent_classification"


@pytest.mark.asyncio
async def test_tier1_no_api_call():
    """Tier 1 matches should NOT make any OpenAI API call."""
    mock_client = MagicMock()
    result = await classify_intent("Hello!", openai_client=mock_client)
    assert result["intent"] == "greeting"
    # The mock client should never have been called
    mock_client.chat.completions.create.assert_not_called()


# ── Helpers ─────────────────────────────────────────────────────────────

def _make_mock_client(response_dict: dict) -> MagicMock:
    """Create a mock OpenAI client that returns a specific JSON response."""
    return _make_mock_client_raw(json.dumps(response_dict))


def _make_mock_client_raw(raw_content: str) -> MagicMock:
    """Create a mock AsyncOpenAI client that returns raw string content."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = raw_content
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    return mock_client