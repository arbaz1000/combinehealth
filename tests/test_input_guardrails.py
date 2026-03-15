"""
Tests for input guardrails (app/guardrails.py — check_input).
"""

from app.guardrails import check_input


# ── Empty / whitespace ────────────────────────────────────────────────────

def test_empty_string_rejected():
    ok, msg = check_input("")
    assert not ok
    assert "enter a question" in msg.lower()


def test_whitespace_only_rejected():
    ok, msg = check_input("   \n\t  ")
    assert not ok
    assert "enter a question" in msg.lower()


def test_none_rejected():
    ok, msg = check_input(None)
    assert not ok


# ── Length limit ──────────────────────────────────────────────────────────

def test_over_2000_chars_rejected():
    ok, msg = check_input("a" * 2001)
    assert not ok
    assert "too long" in msg.lower()


def test_exactly_2000_chars_accepted():
    ok, msg = check_input("a" * 2000)
    assert ok


def test_normal_length_accepted():
    ok, msg = check_input("Is spinal ablation covered under UHC?")
    assert ok


# ── PII detection: SSN ───────────────────────────────────────────────────

def test_ssn_detected():
    ok, msg = check_input("My SSN is 123-45-6789")
    assert not ok
    assert "social security" in msg.lower()


def test_ssn_embedded_in_text():
    ok, msg = check_input("Patient ID 999-88-7777 needs coverage check")
    assert not ok


# ── PII detection: Credit card ───────────────────────────────────────────

def test_credit_card_with_spaces():
    ok, msg = check_input("Card: 4111 1111 1111 1111")
    assert not ok
    assert "credit card" in msg.lower()


def test_credit_card_with_dashes():
    ok, msg = check_input("Pay with 4111-1111-1111-1111")
    assert not ok


def test_credit_card_no_separators():
    ok, msg = check_input("Card number 4111111111111111")
    assert not ok


# ── PII detection: Phone ─────────────────────────────────────────────────

def test_phone_with_dashes():
    ok, msg = check_input("Call me at 555-123-4567")
    assert not ok
    assert "phone" in msg.lower()


def test_phone_with_dots():
    ok, msg = check_input("Reach 555.123.4567")
    assert not ok


# ── PII detection: Email ─────────────────────────────────────────────────

def test_email_detected():
    ok, msg = check_input("Email me at doctor@hospital.com")
    assert not ok
    assert "email" in msg.lower()


# ── Valid medical queries pass ────────────────────────────────────────────

def test_valid_coverage_question():
    ok, _ = check_input("Is spinal ablation covered under UHC commercial plans?")
    assert ok


def test_valid_cpt_question():
    ok, _ = check_input("What CPT codes apply to cardiac catheterization?")
    assert ok


def test_valid_prior_auth_question():
    ok, _ = check_input("Does knee arthroscopy require prior authorization?")
    assert ok


def test_cpt_code_not_flagged_as_credit_card():
    """CPT codes are 5 digits — should NOT trigger credit card detection."""
    ok, _ = check_input("What is CPT 64625?")
    assert ok


def test_policy_number_not_flagged():
    """Policy numbers like 2024T0123 should not trigger PII detection."""
    ok, _ = check_input("Tell me about policy 2024T0538456")
    assert ok