"""
Tests for multi-turn conversation support (Block 3).

Covers:
- History sanitization (consecutive same-role messages, extra fields stripped)
- History truncation (respects MAX_HISTORY_TURNS)
- LLM receives history as native chat messages
- Frontend history format (role + content only)
"""

from unittest.mock import MagicMock, patch, call

from app.rag import sanitize_history, truncate_history, generate_answer


# ── sanitize_history ──────────────────────────────────────────────────


def test_sanitize_empty_history():
    assert sanitize_history(None) == []
    assert sanitize_history([]) == []


def test_sanitize_strips_extra_fields():
    """Only role and content should survive — sources, metadata, etc. are dropped."""
    history = [
        {"role": "user", "content": "Is ablation covered?", "sources": [], "extra": 123},
        {"role": "assistant", "content": "Yes, it is.", "sources": [{"policy": "123"}]},
    ]
    result = sanitize_history(history)
    assert result == [
        {"role": "user", "content": "Is ablation covered?"},
        {"role": "assistant", "content": "Yes, it is."},
    ]


def test_sanitize_merges_consecutive_user_messages():
    """Two consecutive user messages should be merged into one."""
    history = [
        {"role": "user", "content": "Is ablation covered?"},
        {"role": "user", "content": "I mean spinal ablation specifically"},
        {"role": "assistant", "content": "Yes, spinal ablation is covered."},
    ]
    result = sanitize_history(history)
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert "ablation covered?" in result[0]["content"]
    assert "spinal ablation specifically" in result[0]["content"]
    assert result[1]["role"] == "assistant"


def test_sanitize_merges_consecutive_assistant_messages():
    """Two consecutive assistant messages should be merged too."""
    history = [
        {"role": "user", "content": "Tell me about ablation"},
        {"role": "assistant", "content": "Ablation is covered."},
        {"role": "assistant", "content": "Here are the CPT codes..."},
    ]
    result = sanitize_history(history)
    assert len(result) == 2
    assert result[1]["role"] == "assistant"
    assert "Ablation is covered." in result[1]["content"]
    assert "CPT codes" in result[1]["content"]


def test_sanitize_skips_empty_content():
    """Messages with empty/whitespace content should be dropped."""
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": "Hi there!"},
    ]
    result = sanitize_history(history)
    assert len(result) == 2
    assert result[0]["content"] == "Hello"


def test_sanitize_skips_invalid_roles():
    """Messages with roles other than user/assistant should be dropped."""
    history = [
        {"role": "system", "content": "You are a bot"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    result = sanitize_history(history)
    assert len(result) == 2
    assert result[0]["role"] == "user"


def test_sanitize_multiple_consecutive_merges():
    """Three consecutive user messages should all merge into one."""
    history = [
        {"role": "user", "content": "First"},
        {"role": "user", "content": "Second"},
        {"role": "user", "content": "Third"},
        {"role": "assistant", "content": "Response"},
    ]
    result = sanitize_history(history)
    assert len(result) == 2
    assert "First" in result[0]["content"]
    assert "Second" in result[0]["content"]
    assert "Third" in result[0]["content"]


# ── truncate_history ──────────────────────────────────────────────────


def test_truncate_under_limit():
    """History shorter than limit should be returned as-is."""
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
    ]
    result = truncate_history(history, max_turns=5)
    assert result == history


def test_truncate_at_limit():
    """History exactly at limit should be returned as-is."""
    history = []
    for i in range(5):
        history.append({"role": "user", "content": f"Q{i}"})
        history.append({"role": "assistant", "content": f"A{i}"})
    assert len(history) == 10
    result = truncate_history(history, max_turns=5)
    assert result == history


def test_truncate_over_limit():
    """History over limit should keep only the last N turn-pairs."""
    history = []
    for i in range(8):
        history.append({"role": "user", "content": f"Q{i}"})
        history.append({"role": "assistant", "content": f"A{i}"})
    assert len(history) == 16

    result = truncate_history(history, max_turns=3)
    assert len(result) == 6
    # Should keep the last 3 turn-pairs (Q5-A5, Q6-A6, Q7-A7)
    assert result[0] == {"role": "user", "content": "Q5"}
    assert result[-1] == {"role": "assistant", "content": "A7"}


def test_truncate_with_max_turns_1():
    """Max 1 turn-pair should keep only the last 2 messages."""
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    result = truncate_history(history, max_turns=1)
    assert result == [
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]


# ── generate_answer with history ──────────────────────────────────────


def _make_mock_openai_client(answer_text: str = "Mocked answer") -> MagicMock:
    """Create a mock OpenAI client for generate_answer."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = answer_text
    mock_response.usage.prompt_tokens = 200
    mock_response.usage.completion_tokens = 100
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def test_generate_answer_without_history():
    """Without history, messages should be [system, user] only."""
    mock_client = _make_mock_openai_client()

    with patch("app.rag.log_call"):
        generate_answer("Is ablation covered?", "some context", mock_client)

    call_kwargs = mock_client.chat.completions.create.call_args
    messages = call_kwargs[1]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_generate_answer_with_history():
    """With history, messages should be [system, ...history turns..., user]."""
    mock_client = _make_mock_openai_client()

    history = [
        {"role": "user", "content": "Is spinal ablation covered?"},
        {"role": "assistant", "content": "Yes, it is covered under policy 2024T0538."},
    ]

    with patch("app.rag.log_call"):
        generate_answer("What CPT codes apply?", "some context", mock_client, chat_history=history)

    call_kwargs = mock_client.chat.completions.create.call_args
    messages = call_kwargs[1]["messages"]
    assert len(messages) == 4  # system + 2 history + current user
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Is spinal ablation covered?"
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "user"
    assert "CPT codes" in messages[3]["content"]


def test_generate_answer_history_as_native_messages():
    """History should be injected as native role messages, not as text in the user prompt."""
    mock_client = _make_mock_openai_client()

    history = [
        {"role": "user", "content": "Prior question"},
        {"role": "assistant", "content": "Prior answer"},
    ]

    with patch("app.rag.log_call"):
        generate_answer("Follow-up", "context", mock_client, chat_history=history)

    call_kwargs = mock_client.chat.completions.create.call_args
    messages = call_kwargs[1]["messages"]

    # The history messages should be separate message objects, not embedded in user prompt text
    user_prompt_message = messages[-1]
    assert "Prior question" not in user_prompt_message["content"]
    assert "Prior answer" not in user_prompt_message["content"]

    # They should exist as their own messages
    history_messages = messages[1:-1]
    assert len(history_messages) == 2
    assert history_messages[0] == {"role": "user", "content": "Prior question"}
    assert history_messages[1] == {"role": "assistant", "content": "Prior answer"}


# ── ask() integration with history ────────────────────────────────────


def test_ask_sanitizes_and_truncates_history():
    """ask() should sanitize and truncate history before passing to classifier and LLM."""
    with patch("app.rag.classify_intent") as mock_classify, \
         patch("app.rag.retrieve") as mock_retrieve, \
         patch("app.rag.generate_answer") as mock_generate, \
         patch("app.rag.get_openai_client") as mock_oai, \
         patch("app.rag.get_qdrant_client") as mock_qd:

        mock_classify.return_value = {
            "intent": "policy_query",
            "response": None,
            "rewritten_query": "Is ablation covered?",
        }
        mock_retrieve.return_value = [
            {"text": "chunk", "policy_name": "P1", "policy_number": "001",
             "section_name": "S1", "source_url": "http://example.com", "score": 0.9}
        ]
        mock_generate.return_value = "Answer"

        # Send history with consecutive user messages and extra fields
        messy_history = [
            {"role": "user", "content": "First question", "sources": []},
            {"role": "user", "content": "Actually, let me rephrase"},
            {"role": "assistant", "content": "Sure, go ahead.", "sources": [{"p": "1"}]},
        ]

        from app.rag import ask
        ask("Is ablation covered?", chat_history=messy_history)

        # Classifier should receive sanitized history
        classify_call = mock_classify.call_args
        passed_history = classify_call[0][1]  # second positional arg
        # Consecutive user messages should be merged
        assert len(passed_history) == 2
        assert passed_history[0]["role"] == "user"
        assert "First question" in passed_history[0]["content"]
        assert "rephrase" in passed_history[0]["content"]

        # generate_answer should also receive sanitized history
        gen_call = mock_generate.call_args
        gen_history = gen_call[1]["chat_history"]
        assert len(gen_history) == 2