"""
Tests for output guardrails (app/guardrails.py — check_output).

Tests use synthetic answers and chunk dicts to verify:
  1. Medical disclaimer appended to policy answers
  2. Hallucination flag when answer cites unknown policy numbers
  3. Off-topic leak detection for prescriptive/diagnosis/legal/financial language
"""

from app.guardrails import (
    check_output,
    MEDICAL_DISCLAIMER,
    HALLUCINATION_WARNING,
    OFFTOPIC_WARNING,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _chunk(policy_number: str = "2024T0538456") -> dict:
    """Create a minimal chunk dict with a given policy number."""
    return {
        "text": "sample text",
        "policy_name": "Test Policy",
        "policy_number": policy_number,
        "section_name": "Coverage",
        "source_url": "https://example.com",
        "score": 0.85,
    }


# ── Medical disclaimer ──────────────────────────────────────────────────

def test_disclaimer_appended():
    result = check_output("Spinal ablation is covered.", [_chunk()])
    assert result.endswith(MEDICAL_DISCLAIMER)


def test_disclaimer_separated_by_double_newline():
    result = check_output("Spinal ablation is covered.", [_chunk()])
    assert "\n\n*This information" in result


def test_disclaimer_is_italic():
    """Disclaimer text is wrapped in markdown italic markers."""
    assert MEDICAL_DISCLAIMER.strip().startswith("*")
    assert MEDICAL_DISCLAIMER.strip().endswith("*")


def test_disclaimer_on_empty_chunks():
    """Disclaimer still appended even when no chunks (none-tier LLM call)."""
    result = check_output("No matching policies found.", [])
    assert MEDICAL_DISCLAIMER in result


# ── Hallucination flag ──────────────────────────────────────────────────

def test_no_hallucination_when_policy_matches():
    answer = "According to policy 2024T0538456, this is covered."
    chunks = [_chunk("2024T0538456")]
    result = check_output(answer, chunks)
    assert HALLUCINATION_WARNING not in result


def test_hallucination_flagged_when_policy_not_in_chunks():
    answer = "According to policy 2024T9999999, this is covered."
    chunks = [_chunk("2024T0538456")]
    result = check_output(answer, chunks)
    assert HALLUCINATION_WARNING in result


def test_hallucination_multiple_policies_one_unknown():
    answer = "Policy 2024T0538456 covers this. See also 2023T0000001."
    chunks = [_chunk("2024T0538456")]
    result = check_output(answer, chunks)
    assert HALLUCINATION_WARNING in result


def test_hallucination_all_policies_known():
    answer = "Policy 2024T0538456 and 2023T0612345 both apply."
    chunks = [_chunk("2024T0538456"), _chunk("2023T0612345")]
    result = check_output(answer, chunks)
    assert HALLUCINATION_WARNING not in result


def test_no_hallucination_when_no_policy_numbers_in_answer():
    answer = "This procedure is generally covered under UHC plans."
    chunks = [_chunk()]
    result = check_output(answer, chunks)
    assert HALLUCINATION_WARNING not in result


def test_hallucination_with_empty_chunks():
    """Policy number in answer but no chunks at all → hallucination."""
    answer = "According to policy 2024T0538456, this is covered."
    result = check_output(answer, [])
    assert HALLUCINATION_WARNING in result


# ── Off-topic leak detection ────────────────────────────────────────────

def test_no_offtopic_on_clean_answer():
    answer = "Spinal ablation is covered under policy 2024T0538456."
    result = check_output(answer, [_chunk()])
    assert OFFTOPIC_WARNING not in result


def test_offtopic_diagnosis_you_have():
    answer = "Based on this, you likely have a herniated disc."
    result = check_output(answer, [_chunk()])
    assert OFFTOPIC_WARNING in result


def test_offtopic_diagnosis_i_diagnose():
    answer = "I would diagnose this as a chronic condition."
    result = check_output(answer, [_chunk()])
    assert OFFTOPIC_WARNING in result


def test_offtopic_prescriptive_you_should_see():
    answer = "You should see a specialist for this."
    result = check_output(answer, [_chunk()])
    assert OFFTOPIC_WARNING in result


def test_offtopic_prescriptive_seek_medical():
    answer = "Please seek immediate medical attention for this."
    result = check_output(answer, [_chunk()])
    assert OFFTOPIC_WARNING in result


def test_offtopic_prescriptive_i_recommend():
    answer = "I recommend you start physical therapy immediately."
    result = check_output(answer, [_chunk()])
    assert OFFTOPIC_WARNING in result


def test_offtopic_prescriptive_you_need_to():
    answer = "You need to see a cardiologist about this."
    result = check_output(answer, [_chunk()])
    assert OFFTOPIC_WARNING in result


def test_offtopic_financial_invest():
    answer = "You should invest in a health savings account."
    result = check_output(answer, [_chunk()])
    assert OFFTOPIC_WARNING in result


def test_offtopic_financial_portfolio():
    answer = "Your portfolio should include health stocks."
    result = check_output(answer, [_chunk()])
    assert OFFTOPIC_WARNING in result


def test_offtopic_legal_advice():
    answer = "This is not legal advice, but you could sue."
    result = check_output(answer, [_chunk()])
    assert OFFTOPIC_WARNING in result


def test_offtopic_legal_hire_attorney():
    answer = "You should hire an attorney to handle this claim."
    result = check_output(answer, [_chunk()])
    assert OFFTOPIC_WARNING in result


def test_offtopic_legal_lawsuit():
    answer = "You could file a lawsuit against the insurer."
    result = check_output(answer, [_chunk()])
    assert OFFTOPIC_WARNING in result


def test_offtopic_this_is_my_recommendation():
    answer = "This would be my recommendation for your case."
    result = check_output(answer, [_chunk()])
    assert OFFTOPIC_WARNING in result


# ── Combined warnings ───────────────────────────────────────────────────

def test_hallucination_and_offtopic_both_flagged():
    answer = "Policy 2024T9999999 says you likely have cancer."
    chunks = [_chunk("2024T0538456")]
    result = check_output(answer, chunks)
    assert HALLUCINATION_WARNING in result
    assert OFFTOPIC_WARNING in result
    assert MEDICAL_DISCLAIMER in result


def test_warning_order_hallucination_then_offtopic_then_disclaimer():
    """Warnings appear in order: hallucination, off-topic, disclaimer."""
    answer = "Policy 2024T9999999 says you likely have cancer."
    chunks = [_chunk("2024T0538456")]
    result = check_output(answer, chunks)
    hall_pos = result.index(HALLUCINATION_WARNING)
    offtopic_pos = result.index(OFFTOPIC_WARNING)
    disc_pos = result.index(MEDICAL_DISCLAIMER)
    assert hall_pos < offtopic_pos < disc_pos


# ── Case insensitivity ──────────────────────────────────────────────────

def test_offtopic_case_insensitive():
    answer = "YOU SHOULD SEE a doctor about this."
    result = check_output(answer, [_chunk()])
    assert OFFTOPIC_WARNING in result


# ── Safe phrases that should NOT trigger off-topic ──────────────────────

def test_coverage_language_not_flagged():
    """Normal policy language should not trigger off-topic warnings."""
    answer = (
        "This procedure requires prior authorization. "
        "The policy covers spinal ablation under CPT 64625. "
        "Coverage is contingent on medical necessity documentation."
    )
    result = check_output(answer, [_chunk()])
    assert OFFTOPIC_WARNING not in result


def test_suggest_rephrasing_not_flagged():
    """LLM suggesting to rephrase or contact UHC should not trigger."""
    answer = (
        "I couldn't find specific policy information for this procedure. "
        "Try rephrasing your question with the CPT code, or contact UHC directly."
    )
    result = check_output(answer, [_chunk()])
    assert OFFTOPIC_WARNING not in result