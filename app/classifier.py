"""
Intent classification + query rewriting — decides what to do with each user message.

Two-tier architecture (optimized for latency):
  Tier 1: Regex catches obvious greetings/thanks → instant response, no API call
  Tier 2: GPT-4o-mini classifies ambiguous queries + rewrites follow-ups
          into standalone questions using chat history

Intents:
  greeting     → friendly response, skip retrieval
  off_topic    → polite redirect, skip retrieval
  policy_query → pass through to RAG pipeline as-is
  follow_up    → rewrite into standalone query, then RAG pipeline
"""

import json
import re

from openai import AsyncOpenAI

from app.config import LLM_MODEL, INSURER_NAME, INSURER_SHORT_NAME
from app.cost_tracker import log_call

# ── Tier 1: Regex patterns for obvious intents ─────────────────────────
# These are cheap (no API call) and catch the most common non-policy messages.
# Patterns are intentionally conservative — when in doubt, fall through to Tier 2.

_GREETING_PATTERNS = re.compile(
    r"^\s*("
    r"h(i|ello|ey|owdy|ola)"
    r"|good\s*(morning|afternoon|evening|day)"
    r"|what'?s\s*up"
    r"|yo\b"
    r"|sup\b"
    r")\s*[!?.]*\s*$",
    re.IGNORECASE,
)

_THANKS_PATTERNS = re.compile(
    r"^\s*("
    r"thank(s|\s*you)"
    r"|thx"
    r"|ty\b"
    r"|appreciate\s*it"
    r"|got\s*it"
    r"|perfect"
    r"|great,?\s*thanks?"
    r")\s*[!?.]*\s*$",
    re.IGNORECASE,
)

_BYE_PATTERNS = re.compile(
    r"^\s*("
    r"bye"
    r"|goodbye"
    r"|see\s*ya"
    r"|take\s*care"
    r"|have\s*a\s*good\s*(one|day|night)"
    r")\s*[!?.]*\s*$",
    re.IGNORECASE,
)

# Tier 1 canned responses
_GREETING_RESPONSE = (
    f"Hello! I'm the {INSURER_SHORT_NAME} policy assistant. I can help you understand "
    "coverage details, CPT codes, prior authorization requirements, "
    f"and other {INSURER_NAME} commercial policy questions. "
    "What would you like to know?"
)

_THANKS_RESPONSE = (
    "You're welcome! Let me know if you have any other policy questions."
)

_BYE_RESPONSE = (
    "Goodbye! Feel free to come back anytime you have policy questions."
)


# ── Tier 2: LLM-based classification prompt ────────────────────────────

_CLASSIFIER_SYSTEM_PROMPT = f"""\
You are an intent classifier for a {INSURER_NAME} ({INSURER_SHORT_NAME}) insurance policy chatbot.

Given the user's message and optional chat history, determine the intent and — if \
needed — rewrite the query into a standalone question.

## Intents
- greeting: casual hellos, greetings, or pleasantries
- off_topic: questions unrelated to {INSURER_SHORT_NAME} insurance policies \
  (weather, sports, coding, general knowledge, etc.)
- policy_query: a direct, self-contained question about {INSURER_SHORT_NAME} insurance policies, \
  coverage, CPT codes, prior auth, etc.
- follow_up: a message that references previous conversation context \
  (pronouns like "it", "that", "those", or phrases like "what about", \
  "and for", "how about") — must be rewritten into a standalone question

## Rules
1. If intent is "follow_up", rewrite the query into a complete standalone question \
   using context from chat_history. The rewritten query must be understandable \
   WITHOUT the chat history.
2. If intent is "policy_query", set rewritten_query to the original message (no change).
3. If intent is "greeting" or "off_topic", generate a short, friendly response.
4. For "off_topic", always gently redirect toward insurance policy questions.
5. When unsure between off_topic and policy_query, prefer policy_query \
   (let the retrieval system handle it).

## Output format (strict JSON)
{{
  "intent": "greeting" | "off_topic" | "policy_query" | "follow_up",
  "response": "direct response text (for greeting/off_topic only, null otherwise)",
  "rewritten_query": "standalone query (for policy_query/follow_up only, null otherwise)"
}}\
"""


def _build_classifier_messages(
    message: str, chat_history: list[dict] | None = None
) -> list[dict]:
    """Build the messages list for the classifier LLM call."""
    history_text = "No prior conversation."
    if chat_history:
        pairs = []
        # Last 5 turn-pairs (10 messages). History is already truncated
        # by rag.truncate_history() before reaching here, but we cap again
        # defensively. See docs/design-decisions.md DD-3 for rationale.
        for msg in chat_history[-10:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            pairs.append(f"{role}: {content}")
        history_text = "\n".join(pairs)

    user_content = (
        f"Chat history:\n{history_text}\n\n"
        f"Current message: {message}"
    )

    return [
        {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_classifier_response(raw: str) -> dict:
    """Parse the JSON response from the classifier LLM. Handles edge cases."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: treat as policy_query to avoid blocking the user
        return {
            "intent": "policy_query",
            "response": None,
            "rewritten_query": raw.strip() or None,
        }

    return {
        "intent": parsed.get("intent", "policy_query"),
        "response": parsed.get("response"),
        "rewritten_query": parsed.get("rewritten_query"),
    }


# ── Public API ──────────────────────────────────────────────────────────


async def classify_intent(
    message: str,
    chat_history: list[dict] | None = None,
    openai_client: AsyncOpenAI | None = None,
) -> dict:
    """
    Classify user intent and optionally rewrite the query.

    Returns:
        {
            "intent": "greeting" | "off_topic" | "policy_query" | "follow_up",
            "response": str | None,        # direct response for greeting/off_topic
            "rewritten_query": str | None,  # standalone query for policy_query/follow_up
        }
    """
    # ── Tier 1: regex for obvious patterns (no API call) ────────────
    if _GREETING_PATTERNS.match(message):
        return {
            "intent": "greeting",
            "response": _GREETING_RESPONSE,
            "rewritten_query": None,
        }

    if _THANKS_PATTERNS.match(message):
        return {
            "intent": "greeting",
            "response": _THANKS_RESPONSE,
            "rewritten_query": None,
        }

    if _BYE_PATTERNS.match(message):
        return {
            "intent": "greeting",
            "response": _BYE_RESPONSE,
            "rewritten_query": None,
        }

    # ── Tier 2: LLM classification (GPT-4o-mini, JSON mode) ────────
    if openai_client is None:
        from app.rag import get_openai_client
        openai_client = get_openai_client()

    messages = _build_classifier_messages(message, chat_history)

    response = await openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.0,  # deterministic classification
        max_tokens=256,   # classification output is small
        response_format={"type": "json_object"},
    )

    usage = response.usage
    log_call(
        call_type="chat_completion",
        model=LLM_MODEL,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        metadata={"purpose": "intent_classification"},
    )

    raw = response.choices[0].message.content
    return _parse_classifier_response(raw)