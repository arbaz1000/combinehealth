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

## DD-2: Consecutive Same-Role Messages — Backend Sanitization

**Date:** 2026-03-16

**Decision:** Sanitize chat history on the backend by merging consecutive same-role messages before passing to the classifier or LLM.

**Context:** In the Streamlit frontend, it's possible for a user to send two messages before receiving a response (e.g., via sidebar example buttons). This produces consecutive `user` messages in the history, which confuses turn-pair truncation logic and degrades LLM response quality.

**Approach:**
- **Backend (defense-in-depth):** `sanitize_history()` in `app/rag.py` merges consecutive same-role messages into one. This protects against any client sending malformed history.
- **Frontend (prevention):** Sidebar example buttons are disabled during request processing via a `st.session_state.processing` flag. The main `st.chat_input` is naturally guarded by Streamlit (disables itself during spinner).
- **Stop button deferred to Block 6:** A "stop generation" button (like ChatGPT/Perplexity) only makes sense with streaming. It will be added in Block 6 (Streaming Frontend) when SSE streaming is in place and tokens arrive incrementally.

---

## DD-3: Chat History Truncation — Fixed 5 Turn-Pairs

**Date:** 2026-03-16

**Decision:** Limit chat history to the last 5 turn-pairs (10 messages) for both the classifier and the LLM generation call.

**Context:** Sending the entire conversation to the LLM increases token cost and can degrade response quality when older turns are no longer relevant. We needed a truncation strategy.

**Rationale:**
1. **5 turn-pairs covers most practical follow-up chains.** Insurance policy Q&A rarely requires context beyond the last few exchanges. A user might ask about a procedure, then its CPT codes, then prior auth requirements — that's 3 turns.
2. **Predictable cost.** Fixed truncation keeps per-request token usage bounded. With GPT-4o-mini at ~10 messages of context, we stay well within reasonable cost per query.
3. **Simplicity.** A fixed window is easy to reason about, test, and debug.

**Production extensions (when to revisit):**
- **Token-budget-based truncation:** Instead of counting turns, count tokens and keep as many recent turns as fit within a budget (e.g., 2000 tokens for history). This handles variable-length messages better.
- **Summarization:** Summarize older turns into a compact paragraph, then append recent turns verbatim. This preserves long-range context without blowing up token count. Trade-off: adds one extra LLM call.
- **Sliding window + pinned messages:** Keep the first turn (establishes topic) + last N turns. Useful when the opening question sets context that later turns reference.
- **Configurable per-deployment:** Expose `MAX_HISTORY_TURNS` as an environment variable for production tuning without code changes.

---