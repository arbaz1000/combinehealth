"""
Guardrails — input validation, retrieval quality checks, and output checks.

Input checks (check_input):
  1. Empty/whitespace (backend safety net — frontend also disables send button)
  2. PII detection (SSN, credit card, phone, email via regex → redacted)

Retrieval checks (check_retrieval):
  1. Score threshold — drop chunks below RETRIEVAL_SCORE_THRESHOLD
  2. Confidence tier — "high", "low", or "none" based on best surviving score

Output checks (check_output):
  1. Medical disclaimer — appended to every policy answer (italic, double newline)
  2. Hallucination flag — policy numbers in answer not present in retrieved chunks
  3. Off-topic leak detection — prescriptive/diagnosis/legal/financial language

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


# ── Output guardrails ──────────────────────────────────────────────────

MEDICAL_DISCLAIMER = (
    "\n\n*This information is for reference only. "
    "Verify with UHC directly before making coverage decisions.*"
)

# Regex to extract policy numbers from LLM answers.
# UHC policy numbers follow patterns like 2024T0538456, 2023T0612, etc.
POLICY_NUMBER_PATTERN = re.compile(r"\b(\d{4}T\d{4,})\b")

# ── Off-topic leak patterns ────────────────────────────────────────────
# Curated trigger phrases for detecting when GPT-4o-mini drifts into
# prescriptive, diagnostic, legal, or financial advice territory.
# GPT-4o-mini tends to be more helpful and less conservative than GPT-4,
# so this is a warranted lightweight check. See docs/design-decisions.md DD-5.

OFF_TOPIC_PATTERNS = [
    # Medical diagnosis / prescriptive language
    re.compile(r"\byou (?:likely |probably |may )?have\b", re.IGNORECASE),
    re.compile(r"\bI (?:would )?diagnose\b", re.IGNORECASE),
    re.compile(r"\byou should (?:see|visit|consult|seek|go to)\b", re.IGNORECASE),
    re.compile(r"\bseek (?:immediate |emergency )?medical (?:attention|help|care)\b", re.IGNORECASE),
    re.compile(r"\bthis (?:would be |is )my (?:recommendation|diagnosis)\b", re.IGNORECASE),
    re.compile(r"\bI recommend (?:you |that you )?(?:take|start|stop|avoid|try)\b", re.IGNORECASE),
    re.compile(r"\byou need to (?:take|start|stop|see|visit)\b", re.IGNORECASE),
    # Financial advice
    re.compile(r"\b(?:you should |I recommend you )?invest\b", re.IGNORECASE),
    re.compile(r"\byour (?:stock|portfolio|retirement|savings)\b", re.IGNORECASE),
    re.compile(r"\bfinancial advice\b", re.IGNORECASE),
    # Legal advice
    re.compile(r"\blegal advice\b", re.IGNORECASE),
    re.compile(r"\b(?:you should |I recommend you )?(?:hire|consult|retain) an? (?:attorney|lawyer)\b", re.IGNORECASE),
    re.compile(r"\byou (?:could |should |may want to )?(?:file a |bring a )?lawsuit\b", re.IGNORECASE),
    re.compile(r"\byou have (?:a |the )?(?:legal |)right to sue\b", re.IGNORECASE),
]

OFFTOPIC_WARNING = (
    "\n\n*Warning: This response may contain advice outside the scope of "
    "insurance policy information. Please consult the appropriate professional.*"
)

HALLUCINATION_WARNING = (
    "\n\n*Note: This response references policy number(s) not found in the "
    "retrieved sources. Please verify independently.*"
)


def check_output(answer: str, chunks: list[dict]) -> str:
    """
    Post-process LLM answer: append disclaimer, flag hallucinated policy
    numbers, and detect off-topic drift.

    Applied only to policy answers (greeting/off_topic skip this).

    Returns the answer with any appended warnings/disclaimer.
    """
    warnings = []

    # 1. Hallucination flag — policy numbers in answer not in retrieved chunks
    retrieved_policy_numbers = {
        c.get("policy_number", "") for c in chunks if c.get("policy_number")
    }
    mentioned_policy_numbers = set(POLICY_NUMBER_PATTERN.findall(answer))
    hallucinated = mentioned_policy_numbers - retrieved_policy_numbers
    if hallucinated:
        warnings.append(HALLUCINATION_WARNING)

    # 2. Off-topic leak detection
    for pattern in OFF_TOPIC_PATTERNS:
        if pattern.search(answer):
            warnings.append(OFFTOPIC_WARNING)
            break  # one warning is enough

    # 3. Medical disclaimer — always last (skip if LLM already included it)
    if MEDICAL_DISCLAIMER.strip() not in answer:
        warnings.append(MEDICAL_DISCLAIMER)

    return answer + "".join(warnings)