# Design Decisions

This document captures product and design reasoning behind decisions that aren't purely technical architecture choices (those live in `architecture-decisions.md`).

---

## DD-1: Prompt Injection Detection — Intentionally Skipped

**Date:** 2026-03-16

**Decision:** Do not implement prompt injection detection in input guardrails.

**Context:** When designing the input guardrail layer (Block 1), we evaluated whether to add prompt injection detection (blocklist of phrases like "ignore previous instructions", "act as", etc.).

**Rationale:**

1. **Trusted user base.** The chatbot is built for doctors and clinic staff querying insurance coverage. These are internal, professional users with no incentive to jailbreak a policy lookup tool. This is fundamentally different from a public-facing consumer chatbot where adversarial inputs are expected.

2. **False-positive risk on medical queries.** Insurance and medical language naturally contains phrases that overlap with injection patterns. For example:
   - "ignore prior authorization requirements" — legitimate coverage question
   - "act as if the patient has already met the deductible" — valid hypothetical
   - "forget the previous policy and look at the 2025 update" — valid multi-turn refinement

   A blocklist approach would reject these, degrading UX for the exact users we're serving.

3. **Output guardrails are the right layer.** Even if a user somehow coerces the LLM into generating off-topic content, the output guardrails (Block 7) catch responses containing financial advice, legal advice, or medical diagnoses. This is a more robust defense because it catches the *effect* (bad output) regardless of the *cause* (injection, hallucination, or ambiguous query).

4. **Cost of sophistication.** A classifier-based approach (using an LLM call to detect injection) would add latency and cost to every request. For this user base and threat model, the ROI is negative.

**If this changes:** If the chatbot is ever exposed to untrusted public users, revisit this decision. At that point, consider:
- A lightweight classifier (e.g., fine-tuned DistilBERT on injection datasets)
- Prompt hardening techniques (sandwich defense, XML tagging)
- Rate limiting per session as a complementary control

---