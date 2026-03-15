# CombineHealth — UHC Insurance Policy RAG Chatbot

A production-grade RAG (Retrieval-Augmented Generation) chatbot that helps doctors and clinic staff query UnitedHealthcare (UHC) commercial medical policies. Ask about coverage, CPT codes, prior authorization requirements, and more — get sourced, structured answers streamed in real time.

## Key Features

- **Intent Classification** — Two-tier system (regex + GPT-4o-mini) routes greetings and off-topic queries instantly, saving retrieval costs
- **Multi-Turn Conversation** — Follow-up questions are rewritten into standalone queries using chat history
- **Streaming Responses (SSE)** — Tokens stream to the UI as they're generated, with a blinking cursor effect
- **Input Guardrails** — PII detection with redaction (SSN, credit card, phone, email); empty input rejection
- **Retrieval Guardrails** — Cosine similarity filtering with three confidence tiers driving different LLM prompts
- **Output Guardrails** — Medical disclaimer, hallucinated policy number detection, off-topic drift warnings
- **Structured Answers** — Coverage Summary → Requirements → Codes → Prior Auth → Related Policies
- **Cost Tracking** — Every OpenAI API call logged with token counts and USD cost

## Architecture

### High-Level: Query Pipeline

```
User Message
    │
    ├─ 1. Input Guardrails ──→ reject empty, redact PII
    │
    ├─ 2. Intent Classification
    │      Tier 1: regex (greeting/thanks/bye → instant response)
    │      Tier 2: GPT-4o-mini (off_topic / policy_query / follow_up)
    │      ↓ greeting/off_topic → stream canned response, skip retrieval
    │
    ├─ 3. Query Rewriting (for follow-ups)
    │      "What about lumbar?" + history → "Is lumbar spinal ablation covered under UHC?"
    │
    ├─ 4. Retrieve ──→ embed query (OpenAI), search Qdrant (top-8 chunks)
    │
    ├─ 5. Retrieval Guardrails ──→ score filtering + confidence tier
    │      high (≥0.5) │ low (<0.5) │ none (all <0.35)
    │
    ├─ 6. Streaming Generation (SSE) ──→ GPT-4o-mini with tier-specific prompt
    │
    └─ 7. Output Guardrails ──→ disclaimer, hallucination flag, off-topic warning
```

### Low-Level: Component Map

```
app/
├── api.py           FastAPI backend — POST /ask, POST /ask/stream, GET /health, GET /costs
├── rag.py           Core pipeline — ask() (JSON) and ask_stream() (SSE generator)
├── classifier.py    Two-tier intent classification + query rewriting
├── guardrails.py    Input validation, PII redaction, retrieval filtering, output checks
├── config.py        Centralized settings (env vars, thresholds, model config)
├── cost_tracker.py  Per-call OpenAI cost logging (JSONL + summary JSON)
└── frontend.py      Streamlit chat UI — SSE streaming, sources, retry on error

scripts/
├── scrape_policies.py   Scrape UHC policy PDFs from the web
├── parse_and_chunk.py   Parse PDFs with Docling, chunk into passages
└── embed_and_index.py   Embed chunks with fastembed, index in Qdrant

tests/
├── test_input_guardrails.py
├── test_classifier.py
├── test_multi_turn.py
├── test_retrieval_guardrails.py
├── test_streaming.py
├── test_streaming_frontend.py
├── test_output_guardrails.py
└── test_enhanced_prompt.py

config/insurers/uhc.yaml   Insurer-specific scraping config
docs/
├── architecture-decisions.md   ADRs (retrieval, chunking, embedding, streaming, async)
└── design-decisions.md         Product/design rationale (guardrails, prompting, UX)
```

### SSE Event Protocol

All response types (greeting, off-topic, policy answers) flow through the same SSE interface:

```
1. {"type": "intent",  "intent": "policy_query", "rewritten_query": "..."}
2. {"type": "token",   "content": "..."}    ← one per token/word
3. {"type": "sources", "sources": [...]}
4. {"type": "done"}
```

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | GPT-4o-mini (via AsyncOpenAI) |
| Embeddings (query) | OpenAI text-embedding-3-small (1536-dim) |
| Embeddings (index) | BAAI/bge-small-en-v1.5 via fastembed (384-dim, local) |
| Vector Store | Qdrant (local persistent mode, hybrid dense+sparse) |
| Backend | FastAPI + uvicorn |
| Frontend | Streamlit |
| PDF Parsing | Docling |
| Testing | pytest + pytest-asyncio |

## Setup

### Prerequisites

- Python 3.12+
- An OpenAI API key

### 1. Clone and install

```bash
git clone <repo-url>
cd combinehealth
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

### 3. Data pipeline (one-time)

Run these in order to build the vector index:

```bash
# 1. Scrape UHC policy PDFs
python -m scripts.scrape_policies

# 2. Parse PDFs and chunk into passages
python -m scripts.parse_and_chunk

# 3. Embed chunks and index in Qdrant
python -m scripts.embed_and_index
```

This produces ~9,136 chunks from 253 UHC policy PDFs, stored in `data/qdrant_store/`.

### 4. Run the application

Start both servers (two terminals):

```bash
# Terminal 1: FastAPI backend
uvicorn app.api:app --reload

# Terminal 2: Streamlit frontend
streamlit run app/frontend.py
```

The chatbot UI opens at `http://localhost:8501`. The API is at `http://localhost:8000`.

## Usage

### Chat UI

Type a question in the chat input or click an example in the sidebar:

- **Policy query:** "Is spinal ablation covered under UHC commercial plans?"
- **Follow-up:** "What CPT codes apply?" (automatically rewritten using context)
- **Greeting:** "Hi!" → friendly response, no retrieval
- **Off-topic:** "What's the weather?" → polite redirect

Answers stream token-by-token with sources shown in an expander after completion.

### API

**POST /ask/stream** — Streaming (primary)
```bash
curl -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Is spinal ablation covered?", "chat_history": []}'
```

**POST /ask** — JSON (fallback)
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Is spinal ablation covered?", "chat_history": []}'
```

**GET /health** — Health check (returns Qdrant collection status)

**GET /costs** — OpenAI API cost summary

## Key Design Decisions

Detailed rationale is documented in two files:

- **[docs/architecture-decisions.md](docs/architecture-decisions.md)** — Technical ADRs: retrieval architecture, chunking strategy, embedding choice, streaming protocol, async conversion
- **[docs/design-decisions.md](docs/design-decisions.md)** — Product/design rationale: prompt injection (intentionally skipped), history truncation, retrieval guardrails (tiered prompting), output guardrails (lightweight regex)

Notable decisions:

| Decision | Choice | Why |
|---|---|---|
| Prompt injection detection | Skipped | Trusted user base (doctors/staff), false-positive risk on medical language, output guardrails catch the effect |
| PII handling | Redact, don't reject | User might accidentally paste PII — redact it and process the question |
| Low/no retrieval results | Always call LLM | LLM can suggest rephrasing; canned strings feel jarring. Closed-book enforcement prevents hallucination |
| Output guardrails | Regex, not NLI | Zero latency/cost for an internal tool; NLI adds ~486ms for marginal benefit |
| Streaming | Unified SSE for all intents | Canned responses (greeting/off-topic) stream word-by-word for consistent UX |
| OpenAI calls | AsyncOpenAI + native await | FastAPI is async — sync calls block the event loop, limiting concurrency |

## Configuration

Key settings in `app/config.py`:

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gpt-4o-mini` | LLM for generation and classification |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Query embedding model |
| `TOP_K` | `8` | Chunks retrieved per query |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.35` | Minimum cosine similarity to keep a chunk |
| `RETRIEVAL_LOW_CONFIDENCE_THRESHOLD` | `0.5` | Below this, LLM gets uncertainty caveats |
| `MAX_HISTORY_TURNS` | `5` | Turn-pairs sent to LLM (5 pairs = 10 messages) |
| `LLM_TEMPERATURE` | `0.1` | Low temperature for factual responses |
| `LLM_MAX_TOKENS` | `1024` | Max response length |

## Testing

```bash
pytest tests/ -v
```

Tests use `pytest-asyncio` with `AsyncMock` to mock OpenAI calls. No real API calls or Qdrant connections needed.

## Project Structure

```
combinehealth/
├── app/                    Application code
│   ├── api.py
│   ├── classifier.py
│   ├── config.py
│   ├── cost_tracker.py
│   ├── frontend.py
│   ├── guardrails.py
│   └── rag.py
├── scripts/                Data pipeline scripts
│   ├── scrape_policies.py
│   ├── parse_and_chunk.py
│   └── embed_and_index.py
├── tests/                  Test suite
├── config/insurers/        Insurer-specific configs
│   └── uhc.yaml
├── docs/                   Decision documentation
│   ├── architecture-decisions.md
│   └── design-decisions.md
├── data/                   Generated data (gitignored)
│   ├── pdfs/               Downloaded policy PDFs
│   ├── chunks/             Parsed/chunked text
│   ├── qdrant_store/       Vector index
│   └── costs/              API cost logs
├── requirements.txt
├── .env                    API keys (gitignored)
└── .gitignore
```

## Production Extensions

These are documented throughout the codebase and ADRs as future improvements:

- **Cross-encoder reranking** (Cohere Rerank / bge-reranker) for more accurate retrieval scoring
- **Token-budget history truncation** instead of fixed turn-pair count
- **NLI-based faithfulness checking** for async quality monitoring
- **Presidio or AWS Comprehend** for HIPAA-grade PII detection
- **Evaluation framework** (retrieval recall, LLM-as-judge, multi-turn scenarios)
- **Docker/Kubernetes deployment** with health checks and autoscaling
- **Rate limiting** per session if exposed to untrusted users