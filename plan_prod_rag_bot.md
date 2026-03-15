# Production Upgrade Plan — Perplexity-Style RAG Chatbot

## Context

The basic RAG pipeline is complete and working: 253 UHC policy PDFs scraped, parsed into 9,136 chunks, embedded with OpenAI text-embedding-3-small, indexed in Qdrant, served via FastAPI + Streamlit. The chatbot answers questions but lacks production-grade features: no streaming, single-turn only, no guardrails, greetings trigger retrieval unnecessarily.

**Goal:** Upgrade to a production-grade Perplexity-style chatbot with streaming, multi-turn conversation, intent classification, guardrails, and an evaluation framework. Each feature block is committed separately with descriptive messages.

---

## Workflow: Per-Block Approval + Cross-Chat Handoff

### Per-Block Process
1. **Before implementing:** Share the block details with user, ask if any changes are needed
2. **Implement** the block
3. **After implementing:** Show what was done (files changed, key decisions), ask user to review
4. **Commit** after user approval
5. **End of chat:** Provide a handoff prompt for the next block (see below)

### Cross-Chat Handoff
Each block is implemented in a **separate chat** to avoid context limits. At the end of each chat, the assistant generates a **handoff prompt** the user copies into a new chat. The prompt contains:
- Which block was just completed
- Which block to implement next
- Pointer to this plan file: `/Users/apple/.claude/plans/temporal-knitting-globe.md`
- Pointer to memory index: `/Users/apple/.claude/projects/-Users-apple-arbaz-combinehealth/memory/MEMORY.md`
- Any deviations from the plan that happened during implementation
- The exact git commit hash of the current state

**Template for handoff prompt (generated at end of each chat):**
```
I'm building a production-grade UHC insurance policy RAG chatbot.

Read the plan at /Users/apple/.claude/plans/temporal-knitting-globe.md and memory at /Users/apple/.claude/projects/-Users-apple-arbaz-combinehealth/memory/MEMORY.md

Current state: Block {N} ("{name}") is complete. Latest commit: {hash}.
Next: Implement Block {N+1} ("{name}") from the plan.

Deviations from plan so far: {none | list}

Before starting, read the plan file and the relevant existing code files, then share what you're about to do and ask if I want any changes.
```

This ensures each new chat has full context without re-explaining everything.

---

## Query Pipeline (Target Architecture)

```
User Message
    │
    ├─ 1. Input Guardrails ──→ reject/sanitize bad input
    │
    ├─ 2. Intent Classification ──→ greeting | off-topic | policy_query
    │      (if greeting/off-topic → respond directly, skip retrieval)
    │
    ├─ 3. Query Rewriting ──→ resolve multi-turn references
    │      ("What about lumbar?" + history → "Is lumbar spinal ablation covered under UHC?")
    │
    ├─ 4. Retrieve (existing) ──→ Qdrant vector search, top-8 chunks
    │
    ├─ 5. Retrieval Guardrails ──→ score threshold, min-chunk filter
    │
    ├─ 6. Streaming Generation ──→ SSE token-by-token to frontend
    │
    └─ 7. Output Guardrails ──→ medical disclaimer, hallucination flags
```

---

## Implementation Blocks (10 commits)

### Block 0: Initialize Git Repo
- `git init`, commit all current working files
- Verify `.gitignore` excludes `data/`, `.env`, `__pycache__/`
- Commit message: `"Initial commit: working RAG chatbot with scraping, parsing, embedding, and basic Q&A"`

### Block 1: Input Guardrails (`app/guardrails.py`)
- **New file:** `app/guardrails.py` with `check_input(text) -> (ok, message)`
- Checks: empty/whitespace, length limit (2000 chars), PII detection (SSN/credit card/phone regex), prompt injection patterns (common "ignore instructions" phrases)
- Wire into `app/api.py` — call before any processing
- **Files:** `app/guardrails.py`, `app/api.py`
- **Test:** Send empty string, 3000-char input, "ignore your instructions and...", SSN-like string → all rejected with user-friendly messages

### Block 2: Intent Classification + Query Rewriting (`app/classifier.py`)
- **New file:** `app/classifier.py`
- `classify_intent(message, chat_history) -> {intent, rewritten_query}`
- Single GPT-4o-mini call with structured output (JSON mode)
- Intents: `greeting`, `off_topic`, `policy_query`, `follow_up`
- For `follow_up`: rewrites query using chat history context (e.g., "What about lumbar?" → full standalone question)
- For `greeting`/`off_topic`: returns a canned-style response, no retrieval
- Cost-tracked via existing `log_call()`
- **Files:** `app/classifier.py`, `app/rag.py` (integrate), `app/api.py`
- **Test:** "Hi there" → greeting response (no sources). "What's the weather?" → off-topic. "Is ablation covered?" → policy_query. Follow-up after ablation question → rewritten standalone query.

### Block 3: Multi-Turn Conversation
- **API change:** `POST /ask` request body adds `chat_history: list[{role, content}]`
- Backend passes last 5 turn-pairs to classifier (for rewriting) and to LLM (for context)
- Frontend stores messages in `st.session_state`, sends history with each request
- **Files:** `app/api.py` (request model), `app/rag.py` (accept history), `app/frontend.py` (send history)
- **Test:** Ask "Is spinal ablation covered?" then "What CPT codes apply?" → second answer uses context from first

### Block 4: Retrieval Guardrails
- Add to `app/guardrails.py`: `check_retrieval(chunks) -> (chunks, confidence)`
- Score threshold: drop chunks below 0.35 cosine similarity
- Low-confidence detection: if best chunk score < 0.5, flag as low confidence → LLM prompt gets "Note: retrieved context may not be highly relevant"
- Min-chunk filter: if 0 chunks pass threshold → return "I couldn't find relevant policy information" without calling LLM (saves cost)
- **Files:** `app/guardrails.py`, `app/rag.py`
- **Test:** Ask about something not in policies (e.g., "Does UHC cover time travel?") → no LLM call, direct "not found" response

### Block 5: Streaming Backend (SSE)
- **New endpoint:** `POST /ask/stream` using `StreamingResponse` with `text/event-stream`
- OpenAI client uses `stream=True` on `chat.completions.create`
- SSE format: `data: {"type": "token", "content": "..."}\n\n` for tokens, `data: {"type": "sources", "sources": [...]}\n\n` at end, `data: {"type": "done"}\n\n`
- Intent/rewrite results sent as initial SSE events so frontend can show status
- Keep existing `POST /ask` as non-streaming fallback
- **Files:** `app/rag.py` (new `ask_stream()` generator), `app/api.py` (new endpoint)
- **Test:** `curl` the SSE endpoint, see tokens arrive incrementally

### Block 6: Streaming Frontend
- Streamlit calls `/ask/stream` with `requests.get(stream=True)`
- Use `st.write_stream()` or manual `st.empty()` + incremental write for token display
- Show "Searching policies..." while retrieval happens, then stream answer
- Sources displayed in expanders after stream completes
- For greeting/off-topic intents: display response directly (no streaming needed)
- **Files:** `app/frontend.py`
- **Test:** Type a question, see tokens appear one by one, sources appear at end

### Block 7: Output Guardrails
- Add to `app/guardrails.py`: `check_output(answer, chunks) -> answer`
- Medical disclaimer: append "This information is for reference only. Verify with UHC directly before making coverage decisions." to every policy answer
- Hallucination flag: if answer mentions a policy number not present in the retrieved chunks' metadata → flag it
- Off-topic leak detection: if answer contains financial advice, legal advice, or diagnosis → append warning
- **Files:** `app/guardrails.py`, `app/rag.py`
- **Test:** Verify disclaimer appears on policy answers but not on greetings

### Block 8: Enhanced System Prompt
- Refine the system prompt in `app/rag.py` for production quality:
  - Structured output format (headers, bullets, code formatting for CPT codes)
  - Explicit instruction to distinguish "covered", "not covered/unproven", "requires prior auth"
  - Cross-policy awareness: "If related policies exist, mention them"
  - Tone: professional but accessible for medical office staff
- **Files:** `app/rag.py`
- **Test:** Compare answer quality before/after on 5 test questions

### Block 9: Evaluation Framework (`scripts/evaluate.py`)
- **New file:** `scripts/evaluate.py`
- **3 tiers:**
  1. **Retrieval quality:** Recall@K, MRR — given known question→policy mappings, does retrieval find the right chunks?
  2. **Answer quality (LLM-as-judge):** GPT-4o-mini scores answers on faithfulness (0-5), completeness (0-5), citation accuracy (0-5) by comparing answer against retrieved chunks
  3. **Multi-turn scenarios:** 5 scripted conversations testing follow-up resolution, greeting handling, off-topic rejection
- **Test set:** 20 manually curated question-answer pairs in `eval/test_cases.json` with expected policy numbers
- Output: JSON report + console summary table
- **Files:** `scripts/evaluate.py`, `eval/test_cases.json`
- **Test:** Run `python scripts/evaluate.py` → see scores printed

### Block 10: Documentation
- `README.md`: project overview, architecture (HLD/LLD diagrams), setup instructions, usage, deployment guide, evaluation results
- `docs/architecture-decisions.md`: update with new ADRs (guardrails, streaming, eval approach)
- **Files:** `README.md`, `docs/architecture-decisions.md`

---

## Key Files Modified/Created

| File | Action | Block |
|---|---|---|
| `app/guardrails.py` | **Create** | 1, 4, 7 |
| `app/classifier.py` | **Create** | 2 |
| `app/rag.py` | Modify | 2, 3, 4, 5, 7, 8 |
| `app/api.py` | Modify | 1, 2, 3, 5 |
| `app/frontend.py` | Modify | 3, 6 |
| `scripts/evaluate.py` | **Create** | 9 |
| `eval/test_cases.json` | **Create** | 9 |
| `README.md` | **Create** | 10 |

---

## Verification

After all blocks are committed:

1. **Start servers:** `uvicorn app.api:app` + `streamlit run app/frontend.py`
2. **Greeting test:** Type "Hi!" → friendly response, no sources shown
3. **Off-topic test:** "What's the weather?" → polite redirect, no sources
4. **Single-turn policy:** "Is spinal ablation covered?" → streamed answer with sources + disclaimer
5. **Multi-turn:** Follow up with "What CPT codes apply?" → answer uses context from previous turn
6. **Guardrail test:** Paste a fake SSN → input rejected. Ask about non-existent procedure → "not found" response
7. **Evaluation:** Run `python scripts/evaluate.py` → all metrics above baseline thresholds
8. **Cost check:** Hit `GET /costs` → verify per-call tracking includes classifier + rewrite calls