"""
Tests for Block 8: Enhanced System Prompt.

Validates that SYSTEM_PROMPT and per-tier user prompt templates contain
the key production-quality instructions. These are structural checks —
they verify the prompt text itself, not LLM output (which would require
API calls and be non-deterministic).
"""

from app.rag import (
    SYSTEM_PROMPT,
    USER_PROMPT_HIGH_CONFIDENCE,
    USER_PROMPT_LOW_CONFIDENCE,
    USER_PROMPT_NO_CONTEXT,
)


# ── System prompt structure ──────────────────────────────────────────


class TestSystemPromptCoverageStatus:
    """System prompt should instruct the LLM to lead with a clear coverage verdict."""

    def test_has_coverage_verdicts(self):
        for verdict in ["**Covered**", "**Not Covered/Unproven**", "**Requires Prior Authorization**", "**Conditional**"]:
            assert verdict in SYSTEM_PROMPT, f"Missing coverage verdict: {verdict}"

    def test_instructs_bold_verdict(self):
        assert "bold" in SYSTEM_PROMPT.lower()


class TestSystemPromptResponseStructure:
    """System prompt should define a consistent response layout."""

    def test_has_coverage_summary_section(self):
        assert "## Coverage Summary" in SYSTEM_PROMPT

    def test_has_requirements_section(self):
        assert "## Requirements & Conditions" in SYSTEM_PROMPT

    def test_has_relevant_codes_section(self):
        assert "## Relevant Codes" in SYSTEM_PROMPT

    def test_has_prior_auth_section(self):
        assert "## Prior Authorization" in SYSTEM_PROMPT

    def test_has_related_policies_section(self):
        assert "## Related Policies" in SYSTEM_PROMPT


class TestSystemPromptFormatting:
    """System prompt should guide markdown formatting for scannable output."""

    def test_instructs_markdown_headers(self):
        assert "markdown" in SYSTEM_PROMPT.lower()

    def test_instructs_bullet_points(self):
        assert "bullet" in SYSTEM_PROMPT.lower()

    def test_mentions_cpt_code_format(self):
        # Should show an example of formatted CPT code with backticks
        assert "`27447`" in SYSTEM_PROMPT or "code` —" in SYSTEM_PROMPT

    def test_scannable_language(self):
        assert "scan" in SYSTEM_PROMPT.lower()


class TestSystemPromptCoreRules:
    """Core safety rules must survive the prompt enhancement."""

    def test_only_use_context(self):
        assert "ONLY answer based on the provided policy context" in SYSTEM_PROMPT

    def test_no_made_up_info(self):
        assert "Never make up coverage information" in SYSTEM_PROMPT

    def test_cite_policy(self):
        assert "policy name" in SYSTEM_PROMPT.lower() and "number" in SYSTEM_PROMPT.lower()

    def test_professional_tone(self):
        assert "professional" in SYSTEM_PROMPT.lower()

    def test_multiple_policy_conflict(self):
        """Should handle cases where multiple policies address the same topic."""
        assert "differ" in SYSTEM_PROMPT.lower() or "conflict" in SYSTEM_PROMPT.lower()


# ── Per-tier user prompt templates ───────────────────────────────────


class TestHighConfidencePrompt:
    """High-confidence prompt should reference the structured response format."""

    def test_references_response_structure(self):
        assert "response structure" in USER_PROMPT_HIGH_CONFIDENCE.lower()

    def test_mentions_coverage_summary(self):
        assert "Coverage Summary" in USER_PROMPT_HIGH_CONFIDENCE

    def test_only_use_context_rule(self):
        assert "ONLY the policy context" in USER_PROMPT_HIGH_CONFIDENCE

    def test_has_context_placeholder(self):
        assert "{context}" in USER_PROMPT_HIGH_CONFIDENCE

    def test_has_question_placeholder(self):
        assert "{question}" in USER_PROMPT_HIGH_CONFIDENCE


class TestLowConfidencePrompt:
    """Low-confidence prompt should emphasize uncertainty and still reference structure."""

    def test_references_response_structure(self):
        assert "response structure" in USER_PROMPT_LOW_CONFIDENCE.lower()

    def test_uncertainty_language(self):
        assert "uncertainty" in USER_PROMPT_LOW_CONFIDENCE.lower()

    def test_only_use_context_rule(self):
        assert "ONLY the policy context" in USER_PROMPT_LOW_CONFIDENCE

    def test_has_context_placeholder(self):
        assert "{context}" in USER_PROMPT_LOW_CONFIDENCE


class TestNoContextPrompt:
    """No-context prompt should guide the user to refine their search."""

    def test_suggests_refinement(self):
        assert "refine" in USER_PROMPT_NO_CONTEXT.lower() or "rephras" in USER_PROMPT_NO_CONTEXT.lower()

    def test_mentions_cpt_codes_as_suggestion(self):
        assert "CPT" in USER_PROMPT_NO_CONTEXT

    def test_no_own_knowledge_rule(self):
        assert "Do NOT answer the question using your own knowledge" in USER_PROMPT_NO_CONTEXT

    def test_has_question_placeholder(self):
        assert "{question}" in USER_PROMPT_NO_CONTEXT