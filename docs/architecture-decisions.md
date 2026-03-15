# Architecture Decisions

This document captures key design decisions, the options we considered, and why we chose what we chose. This is the "decision log" for the project — useful for the README, interviews, and future-you.

---

## ADR-1: Retrieval Architecture — Hybrid RAG (not RAG + SQL)

**Date:** 2026-03-14

**Context:** UHC policy PDFs contain both prose (coverage rationale, clinical evidence) and structured tables (CPT codes, coverage limits). We evaluated six architectures:

| Option | Description | Verdict |
|---|---|---|
| A. Naive RAG | Chunk → embed → retrieve → generate | Too simple; loses table structure |
| B. Hybrid RAG + SQL for tables | Store tables in SQL, prose in vector DB, route at query time | Good in theory, but adds significant complexity (SQL schema, query router, dual retrieval, merge logic) |
| C. Graph RAG | Knowledge graph with entity/relationship extraction | Massive overkill for this dataset; entity extraction on medical policies is error-prone |
| D. MCP-based | Expose policies as MCP tools for agentic LLM | Over-engineered for a chatbot; adds latency from multi-step tool calls |
| E. Agentic RAG with routing | Router LLM classifies query → dispatches to sub-retrievers | Good for complex systems, but adds an extra LLM call per query (cost + latency) |
| F. Fine-tuned / context-stuffed LLM | Fine-tune on policy data or stuff full policies in context | 253 policies × ~30 pages = too large for context; fine-tuning loses traceability |

**Decision:** Hybrid RAG with dense + sparse search (Option A enhanced), keeping tables as markdown within the vector store.

**Why not RAG + SQL (Option B)?**
- The tables in UHC policies are mostly **CPT code lookup tables** and **coverage condition lists** — they don't have the relational structure that SQL excels at (no JOINs, no aggregations needed).
- Docling converts tables to clean markdown format, which embeds well and is readable by the LLM.
- Sparse/BM25 search handles exact code lookups (e.g., "CPT 64625") effectively — this is the main use case where SQL would have helped.
- Adding SQL would require: schema design per insurer, a query router to decide SQL vs vector, SQL query generation (error-prone), and result merging. This roughly doubles implementation complexity for marginal accuracy gains.
- **If we later find that table lookups are inaccurate**, we can add a SQL layer without changing the existing pipeline — the architecture is extensible.

**Trade-off acknowledged:** For queries like "what is the exact coverage limit for procedure X under plan Y", a SQL lookup would be more precise than vector similarity. We accept this trade-off in favor of simplicity and faster delivery.

---

## ADR-2: Chunk Filtering — Drop chunks < 50 characters

**Date:** 2026-03-14

**Context:** After parsing 253 PDFs with Docling, we found 267 out of 9,403 chunks (2.8%) were under 50 characters. Inspection revealed these were:

- Docling image placeholder artifacts: `<!-- image -->\n\nMedical Policy` (30 chars)
- Single-line fragments: `<!-- image -->` (14 chars)
- Orphaned single-bullet items: `- Provider Administered Drugs - Site of Care` (44 chars)

**Decision:** Filter out all chunks with < 50 characters before indexing.

**Rationale:**
- These chunks contain zero useful information for answering policy questions.
- They would pollute search results — a 14-char `<!-- image -->` chunk could match queries about images/imaging and waste a retrieval slot.
- 267 chunks is 2.8% of total — negligible data loss, significant noise reduction.
- No legitimate policy content (coverage rules, CPT codes, clinical evidence) is under 50 characters.

**Result:** 9,136 clean chunks retained from 9,403 total.

---

## ADR-3: Large Table Chunks — Keep intact, do not split

**Date:** 2026-03-14

**Context:** 443 chunks (4.7%) exceed 4,000 characters. The largest is 22,539 chars. Inspection confirmed these are **100% markdown tables** — CPT code tables, coverage condition tables, etc.

**Decision:** Keep large table chunks intact. Do not split them.

**Rationale:**
- Splitting a table mid-row destroys the data — a CPT code separated from its description is useless.
- The chunker splits on paragraph boundaries (`\n\n`), but markdown tables have no paragraph breaks between rows.
- The embedding model (BGE-small) truncates at 512 tokens, so only the first ~2000 chars get embedded. This means retrieval is based on the top rows of the table, but the full table is available in the LLM context. This is acceptable because:
  - Table headers (which describe what's in the table) are at the top
  - The LLM sees the full chunk text, not just the embedded portion
  - Sparse/BM25 search can still match any code anywhere in the table
- If this becomes a problem, we can split tables row-by-row in a future iteration.

---

## ADR-4: Embedding Strategy — Local fastembed (not OpenAI API)

**Date:** 2026-03-14

**Decision:** Use `BAAI/bge-small-en-v1.5` via fastembed (runs locally) instead of OpenAI's `text-embedding-3-small`.

**Rationale:**
- Free — no per-token cost for 9,136 chunks (OpenAI would cost ~$0.02, trivial, but local is simpler)
- No API key dependency for indexing step
- No rate limits or network failures during bulk indexing
- BGE-small is 384-dim, fast, and performs well on retrieval benchmarks
- Combined with BM25 sparse vectors for hybrid search, overall retrieval quality is strong

**Trade-off:** OpenAI's embedding model may be slightly better on domain-specific medical text. If retrieval quality is poor, switching to OpenAI embeddings is a one-line config change.
