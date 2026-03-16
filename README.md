# CombineHealth — Insurance Policy RAG Chatbot

> **Live Demo:** [https://huggingface.co/spaces/student2/combinehealth](https://huggingface.co/spaces/student2/combinehealth)
>
> The app sleeps after inactivity — if you see a loading screen, wait ~30 seconds for it to wake up.

RAG chatbot that helps doctors and clinic staff query UnitedHealthcare commercial medical policies — coverage, CPT codes, prior authorization, and more. Streamed, sourced, structured answers in real time.

---

## How to Use

1. Open the [live demo](https://huggingface.co/spaces/student2/combinehealth)
2. Type a question or click an example in the sidebar
3. Get a streamed answer with sources

**Try these:**

| Query type | Example | What happens |
|---|---|---|
| Policy question | "Is spinal ablation covered?" | Retrieves relevant policies, streams structured answer with sources |
| Follow-up | "What CPT codes apply?" | Rewrites using conversation context, retrieves, answers |
| Greeting | "Hi!" | Instant friendly response, no retrieval |
| Off-topic | "What's the weather?" | Polite redirect to policy questions |
| PII in input | "My SSN is 123-45-6789, is ablation covered?" | SSN redacted, question still answered |

---

## Architecture

### HLD — System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                             │
│              Streamlit Chat UI (:8501)                       │
│         SSE streaming · chat history · sources              │
└──────────────────────┬──────────────────────────────────────┘
                       │ POST /ask/stream (SSE)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     BACKEND (FastAPI :8000)                  │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │   Input      │  │   Intent     │  │    RAG Pipeline    │ │
│  │   Guardrails │─▶│   Classifier │─▶│                    │ │
│  │   (PII,empty)│  │   (2-tier)   │  │ Retrieve → Filter  │ │
│  └─────────────┘  └──────────────┘  │ → Generate → Guard  │ │
│                                      └────────────────────┘ │
│                                        │            │       │
│                                        ▼            ▼       │
│                                  ┌──────────┐ ┌──────────┐  │
│                                  │  Qdrant  │ │ OpenAI   │  │
│                                  │  (vector │ │ GPT-4o-  │  │
│                                  │   store) │ │ mini     │  │
│                                  └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### HLD — Query Pipeline

```
User Message
 │
 ├─ 1. Input Guardrails ──────→ reject empty, redact PII ([REDACTED])
 │
 ├─ 2. Intent Classification
 │     Tier 1: regex ──→ greeting/thanks/bye → stream canned response, DONE
 │     Tier 2: GPT-4o-mini ──→ off_topic (redirect) | policy_query | follow_up
 │
 ├─ 3. Query Rewriting ───────→ follow-ups rewritten to standalone questions
 │     "What about lumbar?" → "Is lumbar spinal ablation covered under UHC?"
 │
 ├─ 4. Retrieval ─────────────→ embed query (OpenAI) → Qdrant search (top-8)
 │
 ├─ 5. Retrieval Guardrails ──→ score filter + confidence tier
 │     high (≥0.5) → normal RAG │ low (<0.5) → uncertainty caveats │ none → no-context prompt
 │
 ├─ 6. Streaming Generation ──→ GPT-4o-mini with tier-specific prompt → SSE tokens
 │
 └─ 7. Output Guardrails ────→ medical disclaimer · hallucination flag · off-topic warning
```

### LLD — Module Map

```
app/
├── api.py            FastAPI endpoints: POST /ask, POST /ask/stream, GET /health, GET /costs
├── rag.py            Core pipeline: ask() (JSON) + ask_stream() (SSE generator)
│                     System prompt, per-tier user prompts, history management
├── classifier.py     Tier 1 regex + Tier 2 GPT-4o-mini intent classification
│                     Query rewriting for follow-ups using chat history
├── guardrails.py     Input: PII redaction (regex) + empty check
│                     Retrieval: score filtering + confidence tiers
│                     Output: disclaimer + hallucination flag + off-topic detection
├── config.py         All settings from env vars + insurer YAML configs
├── cost_tracker.py   Per-call OpenAI cost logging (JSONL + summary)
└── frontend.py       Streamlit chat UI: SSE parsing, token streaming, sources

scripts/
├── scrape_policies.py    Scrape policy PDFs (configurable per insurer)
├── parse_and_chunk.py    Parse with Docling → section-aware chunks
└── embed_and_index.py    Embed (fastembed) → index in Qdrant

config/insurers/
└── uhc.yaml              Provider-specific: URLs, section names, collection name
```

### LLD — Data Flow (SSE Streaming)

```
Frontend                          Backend                           External
────────                          ───────                           ────────
POST /ask/stream ──────────────▶ check_input()
                                  │
                                 classify_intent() ─────────────▶ OpenAI (Tier 2 only)
                                  │
                            ◀──── SSE: {"type":"intent", ...}
                                  │
                                 embed_query() ─────────────────▶ OpenAI Embeddings
                                 retrieve() ────────────────────▶ Qdrant
                                 check_retrieval()
                                  │
                                 generate_answer_stream() ──────▶ OpenAI Chat (stream=True)
                                  │
                            ◀──── SSE: {"type":"token", ...}     (repeated per token)
                                  │
                                 check_output()
                            ◀──── SSE: {"type":"token", ...}     (guardrail appendix)
                            ◀──── SSE: {"type":"sources", ...}
                            ◀──── SSE: {"type":"done"}
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| LLM | GPT-4o-mini (AsyncOpenAI) | Cheap ($0.15/1M in), fast, strong at RAG |
| Embeddings (query) | OpenAI text-embedding-3-small | High quality, 1536-dim |
| Embeddings (index) | BAAI/bge-small-en-v1.5 (local) | Free, no API dependency for indexing |
| Vector Store | Qdrant (local, hybrid dense+sparse) | BM25 for exact code lookups + dense for semantic |
| Backend | FastAPI + uvicorn | Async-native, SSE streaming support |
| Frontend | Streamlit | Rapid prototyping (production: React/Next.js) |
| PDF Parsing | Docling | Best-in-class table extraction to markdown |
| Testing | pytest + pytest-asyncio | 149 tests, all async, no real API calls |

---

## Setup (Local Development)

```bash
# 1. Clone and install
git clone https://github.com/arbaz1000/combinehealth.git && cd combinehealth
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Build vector index (one-time, ~30 min)
python -m scripts.scrape_policies      # Scrape 253 UHC policy PDFs
python -m scripts.parse_and_chunk      # Parse → 9,136 chunks
python -m scripts.embed_and_index      # Embed → Qdrant index

# 4. Run (two terminals)
uvicorn app.api:app --reload           # Backend → localhost:8000
streamlit run app/frontend.py          # Frontend → localhost:8501
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/ask/stream` | POST | **Primary.** SSE streaming response |
| `/ask` | POST | JSON response (non-streaming fallback) |
| `/health` | GET | Qdrant connection + collection status |
| `/costs` | GET | OpenAI API cost summary |

**Request body** (`/ask` and `/ask/stream`):
```json
{"question": "Is spinal ablation covered?", "chat_history": []}
```

---

## Multi-Provider Extensibility

The system is provider-agnostic. All provider-specific values come from `config/insurers/{name}.yaml`.

**To add a new provider (e.g., Aetna):**

```bash
# 1. Create config
cp config/insurers/uhc.yaml config/insurers/aetna.yaml
# Edit: name, short_name, listing_url, collection_name, etc.

# 2. Run data pipeline for the new provider
INSURER=aetna python -m scripts.scrape_policies
INSURER=aetna python -m scripts.parse_and_chunk
INSURER=aetna python -m scripts.embed_and_index

# 3. Run the app with the new provider
INSURER=aetna uvicorn app.api:app
```

No code changes needed — prompts, UI text, collection names, and guardrail messages all adapt automatically.

---

## Key Design Decisions

Full rationale in [docs/architecture-decisions.md](docs/architecture-decisions.md) and [docs/design-decisions.md](docs/design-decisions.md).

| Decision | Choice | Why |
|---|---|---|
| Retrieval architecture | Hybrid RAG (dense + BM25 sparse) | Tables stay as markdown; BM25 handles exact CPT code lookups |
| Prompt injection | Skipped | Trusted users (doctors); false-positive risk on medical language; output guardrails catch the effect |
| PII handling | Redact, don't reject | User may accidentally paste PII — redact and continue |
| Low/no retrieval | Always call LLM (closed-book) | LLM suggests rephrasing; never uses internal knowledge |
| Output guardrails | Regex, not NLI | Zero cost/latency; NLI adds ~486ms for marginal benefit |
| Streaming | Unified SSE for all intents | Greetings stream word-by-word for consistent UX |
| Async | AsyncOpenAI + native await | Sync calls block the event loop, limiting concurrency |

---

## Production Architecture (AWS)

How this system would scale for production deployment:

```
┌─────────────────────────────────────────────────────────────────────┐
│                           AWS Cloud                                 │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────────┐  │
│  │CloudFront│───▶│   ALB        │───▶│  ECS Fargate              │  │
│  │  (CDN)   │    │(load balance)│    │  ┌────────┐ ┌────────┐   │  │
│  └──────────┘    └──────────────┘    │  │Frontend│ │Backend │   │  │
│                                      │  │(React) │ │(FastAPI│   │  │
│  ┌──────────┐                        │  └────────┘ └───┬────┘   │  │
│  │ Route 53 │                        │                 │        │  │
│  │  (DNS)   │                        └─────────────────┼────────┘  │
│  └──────────┘                                          │           │
│                                           ┌────────────┼────────┐  │
│                                           │            ▼        │  │
│  ┌──────────────┐  ┌──────────────┐  ┌────┴─────┐ ┌────────┐   │  │
│  │ ElastiCache  │  │   RDS        │  │ Qdrant   │ │ OpenAI │   │  │
│  │ (Redis)      │  │ (Postgres)   │  │ Cloud    │ │  API   │   │  │
│  │              │  │              │  │          │ │        │   │  │
│  │ - Response   │  │ - Users      │  │ - Vector │ │ - LLM  │   │  │
│  │   cache      │  │ - Sessions   │  │   index  │ │ - Embed│   │  │
│  │ - Rate limit │  │ - Chat hist  │  │ - Hybrid │ │        │   │  │
│  └──────────────┘  └──────────────┘  └──────────┘ └────────┘   │  │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ CloudWatch   │  │   S3         │  │ Secrets      │              │
│  │ (logs,       │  │ (policy PDFs,│  │ Manager      │              │
│  │  metrics,    │  │  cost logs)  │  │ (API keys)   │              │
│  │  alarms)     │  │              │  │              │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

**Scaling considerations:**

| Concern | Current (Demo) | Production |
|---|---|---|
| Compute | Single container (HF Spaces) | ECS Fargate with auto-scaling (CPU/memory targets) |
| Vector DB | Local Qdrant (file-based) | Qdrant Cloud or self-hosted cluster (replication + sharding) |
| Caching | None | Redis — cache responses by (query_hash, top_chunk_ids) |
| Auth | None | OAuth 2.0 / SSO (clinic staff via existing IdP) |
| Chat persistence | In-memory (Streamlit session) | Postgres — multi-session, cross-device |
| PII detection | Regex patterns | AWS Comprehend Medical or Presidio (HIPAA-grade) |
| Retrieval quality | Cosine similarity only | Cross-encoder reranking (Cohere Rerank / bge-reranker) |
| Monitoring | Cost tracker (JSONL) | CloudWatch metrics + alarms, OpenTelemetry traces |
| Frontend | Streamlit | React/Next.js — custom UI, WebSocket, mobile-responsive |
| Rate limiting | None | Redis token bucket per user/session |
| CI/CD | Manual deploy | GitHub Actions → ECR → ECS rolling deploy |

---

## Testing

```bash
pytest tests/ -v    # 149 tests, ~1.2s, no API calls needed
```

| Test file | Covers |
|---|---|
| `test_input_guardrails.py` | PII redaction, empty input, edge cases |
| `test_classifier.py` | Tier 1 regex, Tier 2 LLM, fallback handling |
| `test_multi_turn.py` | History sanitization, truncation, context injection |
| `test_retrieval_guardrails.py` | Score filtering, confidence tiers, boundary cases |
| `test_streaming.py` | SSE event format, streaming pipeline, all intents |
| `test_streaming_frontend.py` | SSE parsing, token reconstruction |
| `test_output_guardrails.py` | Disclaimer, hallucination flag, off-topic detection |
| `test_enhanced_prompt.py` | System prompt structure, per-tier templates |

---

## Project Structure

```
combinehealth/
├── app/                     Application code (7 modules)
├── scripts/                 Data pipeline (scrape → parse → embed)
├── tests/                   149 tests (pytest-asyncio)
├── config/insurers/         Provider configs (YAML)
├── docs/                    Architecture & design decision logs
├── data/                    Generated data (gitignored)
├── Dockerfile               Multi-stage build for HF Spaces
├── start.sh                 Starts FastAPI + Streamlit
├── requirements.txt         Full dependencies
├── requirements-deploy.txt  Runtime-only dependencies (no docling)
└── .env                     API keys (gitignored)
```