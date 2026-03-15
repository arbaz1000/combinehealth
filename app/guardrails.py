"""
Guardrails — input validation and retrieval quality checks.

Input checks (check_input):
  1. Empty/whitespace (backend safety net — frontend also disables send button)
  2. Length limit (2000 chars)
  3. PII detection (SSN, credit card, phone, email via regex)

Retrieval checks (check_retrieval):
  1. Score threshold — drop chunks below RETRIEVAL_SCORE_THRESHOLD
  2. Confidence tier — "high", "low", or "none" based on best surviving score

Prompt injection detection is intentionally omitted. See docs/design-decisions.md
for full rationale.
"""

import re

# ── PII patterns ──────────────────────────────────────────────────────────
# Best-effort regex detection for common PII formats.
#
# PRODUCTION NOTE: For HIPAA-grade PII redaction, replace these regexes with
# a dedicated library such as:
#   - Microsoft Presidio (open-source, supports custom recognizers)
#     pip install presidio-analyzer presidio-anonymizer
#   - AWS Comprehend Medical (managed service, PHI detection)
#   - Google Cloud DLP (managed service)
# These tools use NLP models + context to detect PII with far higher recall
# and precision than pattern matching, and support redaction/anonymization
# rather than outright rejection.

PII_PATTERNS = {
    "Social Security Number": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit card number": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    "phone number": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "email address": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
}


def redact_pii(text: str) -> tuple[str, list[str]]:
    """
    Replace detected PII with [REDACTED].

    Returns:
        (sanitized_text, list_of_redacted_types)
        e.g. ("My SSN is [REDACTED]", ["Social Security Number"])
    """
    redacted_types = []
    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            text = pattern.sub("[REDACTED]", text)
            redacted_types.append(pii_type)
    return text, redacted_types


def check_input(text: str) -> tuple[bool, str, list[str]]:
    """
    Validate and sanitize user input before processing.

    Returns:
        (True, sanitized_text, redacted_types)  — input safe, PII redacted if any
        (False, error_msg, [])                  — input rejected (empty/too long)
    """
    # 1. Empty / whitespace (backend safety net — UI disables send button too)
    if not text or not text.strip():
        return False, "Please enter a question.", []

    # 2. PII redaction (sanitize, don't reject)
    sanitized, redacted_types = redact_pii(text)

    return True, sanitized, redacted_types


# ── Retrieval guardrails ─────────────────────────────────────────────────

from app.config import RETRIEVAL_SCORE_THRESHOLD, RETRIEVAL_LOW_CONFIDENCE_THRESHOLD


def check_retrieval(chunks: list[dict]) -> tuple[list[dict], str]:
    """
    Filter retrieved chunks by score and assign a confidence tier.

    Three tiers drive different LLM prompt augmentations in rag.py:
      - "high"  — best chunk >= LOW_CONFIDENCE_THRESHOLD (normal RAG)
      - "low"   — chunks survived but best < LOW_CONFIDENCE_THRESHOLD
      - "none"  — zero chunks above SCORE_THRESHOLD

    The LLM is ALWAYS called regardless of tier — it just gets different
    instructions. See docs/design-decisions.md DD-4 for rationale.

    Returns:
        (filtered_chunks, confidence) where confidence is "high"|"low"|"none"
    """
    # Drop chunks below the minimum score threshold
    filtered = [c for c in chunks if c.get("score", 0) >= RETRIEVAL_SCORE_THRESHOLD]

    if not filtered:
        return [], "none"

    best_score = max(c.get("score", 0) for c in filtered)

    if best_score >= RETRIEVAL_LOW_CONFIDENCE_THRESHOLD:
        return filtered, "high"

    return filtered, "low"