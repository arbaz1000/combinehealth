"""
Tests for input guardrails (app/guardrails.py — check_input, redact_pii).
"""

from app.guardrails import check_input, redact_pii


# ── Empty / whitespace ────────────────────────────────────────────────────

def test_empty_string_rejected():
    ok, msg, _ = check_input("")
    assert not ok
    assert "enter a question" in msg.lower()


def test_whitespace_only_rejected():
    ok, msg, _ = check_input("   \n\t  ")
    assert not ok
    assert "enter a question" in msg.lower()


def test_none_rejected():
    ok, msg, _ = check_input(None)
    assert not ok


def test_normal_length_accepted():
    ok, sanitized, _ = check_input("Is spinal ablation covered under UHC?")
    assert ok


# ── PII redaction: SSN ───────────────────────────────────────────────────

def test_ssn_redacted():
    ok, sanitized, redacted = check_input("My SSN is 123-45-6789")
    assert ok
    assert "Social Security Number" in redacted
    assert "[REDACTED]" in sanitized
    assert "123-45-6789" not in sanitized


def test_ssn_embedded_in_text():
    ok, sanitized, redacted = check_input("Patient ID 999-88-7777 needs coverage check")
    assert ok
    assert "[REDACTED]" in sanitized
    assert "999-88-7777" not in sanitized


# ── PII redaction: Credit card ───────────────────────────────────────────

def test_credit_card_with_spaces():
    ok, sanitized, redacted = check_input("Card: 4111 1111 1111 1111")
    assert ok
    assert "credit card number" in redacted
    assert "[REDACTED]" in sanitized
    assert "4111" not in sanitized


def test_credit_card_with_dashes():
    ok, sanitized, redacted = check_input("Pay with 4111-1111-1111-1111")
    assert ok
    assert "[REDACTED]" in sanitized


def test_credit_card_no_separators():
    ok, sanitized, redacted = check_input("Card number 4111111111111111")
    assert ok
    assert "[REDACTED]" in sanitized


# ── PII redaction: Phone ─────────────────────────────────────────────────

def test_phone_with_dashes():
    ok, sanitized, redacted = check_input("Call me at 555-123-4567")
    assert ok
    assert "phone number" in redacted
    assert "[REDACTED]" in sanitized
    assert "555-123-4567" not in sanitized


def test_phone_with_dots():
    ok, sanitized, redacted = check_input("Reach 555.123.4567")
    assert ok
    assert "[REDACTED]" in sanitized


# ── PII redaction: Email ─────────────────────────────────────────────────

def test_email_redacted():
    ok, sanitized, redacted = check_input("Email me at doctor@hospital.com")
    assert ok
    assert "email address" in redacted
    assert "[REDACTED]" in sanitized
    assert "doctor@hospital.com" not in sanitized


# ── Multiple PII types in one message ────────────────────────────────────

def test_multiple_pii_types_redacted():
    ok, sanitized, redacted = check_input("SSN 123-45-6789 email foo@bar.com")
    assert ok
    assert len(redacted) == 2
    assert "123-45-6789" not in sanitized
    assert "foo@bar.com" not in sanitized
    assert sanitized.count("[REDACTED]") == 2


# ── redact_pii standalone ────────────────────────────────────────────────

def test_redact_pii_no_pii():
    text, redacted = redact_pii("Is ablation covered?")
    assert text == "Is ablation covered?"
    assert redacted == []


def test_redact_pii_preserves_surrounding_text():
    text, redacted = redact_pii("My SSN is 123-45-6789 and I need help")
    assert text == "My SSN is [REDACTED] and I need help"
    assert "Social Security Number" in redacted


# ── Valid medical queries pass ────────────────────────────────────────────

def test_valid_coverage_question():
    ok, sanitized, redacted = check_input("Is spinal ablation covered under UHC commercial plans?")
    assert ok
    assert redacted == []
    assert sanitized == "Is spinal ablation covered under UHC commercial plans?"


def test_valid_cpt_question():
    ok, _, redacted = check_input("What CPT codes apply to cardiac catheterization?")
    assert ok
    assert redacted == []


def test_valid_prior_auth_question():
    ok, _, redacted = check_input("Does knee arthroscopy require prior authorization?")
    assert ok
    assert redacted == []


def test_cpt_code_not_flagged_as_credit_card():
    """CPT codes are 5 digits — should NOT trigger credit card detection."""
    ok, _, redacted = check_input("What is CPT 64625?")
    assert ok
    assert redacted == []


def test_policy_number_not_flagged():
    """Policy numbers like 2024T0123 should not trigger PII detection."""
    ok, _, redacted = check_input("Tell me about policy 2024T0538456")
    assert ok
    assert redacted == []