"""
RAG pipeline: retrieve relevant policy chunks from Qdrant, build prompt, call LLM.

Simple mental model:
1. User asks a question
2. We embed the question with OpenAI, search Qdrant for similar chunks
3. We stuff the top results into a prompt with instructions
4. GPT-4o-mini reads the context and answers

All OpenAI calls (embedding + LLM) are cost-tracked automatically.
"""

import json

from openai import OpenAI
from qdrant_client import QdrantClient, models

from app.config import (
    OPENAI_API_KEY,
    QDRANT_PATH,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    TOP_K,
    MAX_HISTORY_TURNS,
)
from app.cost_tracker import log_call
from app.classifier import classify_intent
from app.guardrails import check_retrieval

# ── System prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert insurance policy assistant for doctors and clinic staff.
You help them understand UnitedHealthcare (UHC) commercial medical policies — specifically whether procedures/treatments are covered, what the requirements are, and what codes apply.

RULES:
1. ONLY answer based on the provided policy context. Never make up coverage information.
2. If the context doesn't contain enough information to answer, say so clearly — suggest which policy the user might want to check.
3. Always cite which policy document your answer comes from (policy name + number).
4. When mentioning CPT/HCPCS codes, include the code number AND description.
5. Distinguish clearly between "covered/medically necessary", "not covered/unproven", and "requires prior authorization".
6. Use plain, professional language suitable for medical office staff.
7. If a procedure has specific conditions for coverage (e.g., age requirements, prior treatments needed), list them as bullet points.
8. When asked about a specific procedure or diagnosis, also note any related policies the user might want to review.
"""

# ── Per-tier user prompt templates ────────────────────────────────────
# The LLM is ALWAYS called, but gets different instructions based on
# retrieval confidence. In every tier it is forbidden from using its
# own internal knowledge — answers must come only from retrieved context.

USER_PROMPT_HIGH_CONFIDENCE = """Based on the following UHC policy excerpts, answer the user's question.

--- POLICY CONTEXT ---
{context}
--- END CONTEXT ---

IMPORTANT: Use ONLY the policy context above. Do NOT use your own knowledge about insurance policies, medical procedures, or coverage. If the context does not contain enough information to fully answer, say so — do not fill gaps with outside knowledge.

User question: {question}

Provide a clear, structured answer with policy citations."""

USER_PROMPT_LOW_CONFIDENCE = """The following UHC policy excerpts were retrieved but may not be directly relevant to the question. Use them if they are helpful, but be transparent about uncertainty.

--- POLICY CONTEXT (potentially low relevance) ---
{context}
--- END CONTEXT ---

IMPORTANT:
- Use ONLY the policy context above. Do NOT use your own knowledge about insurance policies, medical procedures, or coverage.
- If the context does not adequately address the question, clearly state that the retrieved policies may not cover this topic.
- Do NOT guess or infer coverage details that are not explicitly stated in the context.
- Suggest that the user rephrase their question or contact UHC directly for clarification.

User question: {question}

Provide what information you can from the context, clearly noting any uncertainty."""

USER_PROMPT_NO_CONTEXT = """No relevant UHC policy excerpts were found for the user's question.

IMPORTANT:
- Do NOT answer the question using your own knowledge. You do not have reliable information about UHC policy coverage beyond what is retrieved from the policy database.
- Explain that no matching policy information was found.
- Suggest the user try rephrasing their question with more specific terms (procedure names, CPT codes, diagnosis codes).
- Recommend contacting UHC directly for coverage questions you cannot answer from the policy database.

User question: {question}

Respond helpfully while making clear you cannot provide policy details without matching context."""


# ── History helpers ────────────────────────────────────────────────────

def sanitize_history(history: list[dict] | None) -> list[dict]:
    """
    Clean up chat history for safe use in LLM messages.

    Handles edge cases like consecutive same-role messages (e.g. user sent
    two messages before getting a response) by merging them into one.
    Strips any extra fields (sources, metadata) — keeps only role + content.
    """
    if not history:
        return []

    cleaned = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role not in ("user", "assistant") or not content.strip():
            continue
        # Merge consecutive same-role messages
        if cleaned and cleaned[-1]["role"] == role:
            cleaned[-1]["content"] += "\n" + content
        else:
            cleaned.append({"role": role, "content": content})
    return cleaned


def truncate_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """
    Keep the last N turn-pairs from history.

    A turn-pair is one user message + one assistant message (2 items).
    Truncates from the front to preserve the most recent context.

    Note: 5 turn-pairs (10 messages) is a pragmatic default that balances
    context quality vs. token cost. In production, consider token-budget-based
    truncation or summarizing older turns. See docs/design-decisions.md DD-3.
    """
    max_messages = max_turns * 2
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(path=QDRANT_PATH)


def embed_query(query: str, client: OpenAI) -> list[float]:
    """Embed a single query string. Cost-tracked."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    log_call(
        call_type="embedding",
        model=EMBEDDING_MODEL,
        input_tokens=response.usage.total_tokens,
        metadata={"purpose": "query_embedding"},
    )
    return response.data[0].embedding


def retrieve(query: str, openai_client: OpenAI, qdrant_client: QdrantClient, top_k: int = TOP_K) -> list[dict]:
    """
    Embed query with OpenAI, then search Qdrant by cosine similarity.
    """
    query_vector = embed_query(query, openai_client)

    results = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
    )

    chunks = []
    for point in results.points:
        payload = point.payload
        chunks.append({
            "text": payload.get("text", ""),
            "policy_name": payload.get("policy_name", ""),
            "policy_number": payload.get("policy_number", ""),
            "section_name": payload.get("section_name", ""),
            "source_url": payload.get("source_url", ""),
            "score": point.score,
        })
    return chunks


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM prompt."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[Source {i}: {chunk['policy_name']} ({chunk['policy_number']}) — Section: {chunk['section_name']}]"
        context_parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(context_parts)


def build_sources(chunks: list[dict]) -> list[dict]:
    """Extract unique source references for citation display."""
    seen = set()
    sources = []
    for chunk in chunks:
        key = chunk["policy_number"]
        if key and key not in seen:
            seen.add(key)
            sources.append({
                "policy_name": chunk["policy_name"],
                "policy_number": chunk["policy_number"],
                "source_url": chunk["source_url"],
            })
    return sources


def generate_answer_stream(
    question: str,
    context: str,
    client: OpenAI,
    chat_history: list[dict] | None = None,
    retrieval_confidence: str = "high",
):
    """
    Stream GPT-4o-mini response token-by-token. Cost-tracked.

    Yields individual token strings as they arrive from the OpenAI API.
    Uses stream_options={"include_usage": True} so the final chunk
    contains token counts for cost tracking (no manual counting needed).
    """
    if retrieval_confidence == "none":
        user_prompt = USER_PROMPT_NO_CONTEXT.format(question=question)
    elif retrieval_confidence == "low":
        user_prompt = USER_PROMPT_LOW_CONFIDENCE.format(context=context, question=question)
    else:
        user_prompt = USER_PROMPT_HIGH_CONFIDENCE.format(context=context, question=question)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": user_prompt})

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        # Final chunk carries usage stats (content is None)
        if chunk.usage is not None:
            log_call(
                call_type="chat_completion",
                model=LLM_MODEL,
                input_tokens=chunk.usage.prompt_tokens,
                output_tokens=chunk.usage.completion_tokens,
                metadata={"question": question[:200], "streaming": True},
            )

        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def generate_answer(
    question: str,
    context: str,
    client: OpenAI,
    chat_history: list[dict] | None = None,
    retrieval_confidence: str = "high",
) -> str:
    """
    Call GPT-4o-mini with the RAG context. Cost-tracked.

    If chat_history is provided, prior turns are injected as native
    user/assistant messages between the system prompt and the current
    RAG prompt. This gives the model natural conversational context
    (vs. stuffing history into the user prompt as flat text).

    retrieval_confidence controls which prompt template is used:
      "high" — normal RAG prompt
      "low"  — adds uncertainty caveats
      "none" — no context, instructs LLM to say it couldn't find info
    """
    if retrieval_confidence == "none":
        user_prompt = USER_PROMPT_NO_CONTEXT.format(question=question)
    elif retrieval_confidence == "low":
        user_prompt = USER_PROMPT_LOW_CONFIDENCE.format(context=context, question=question)
    else:
        user_prompt = USER_PROMPT_HIGH_CONFIDENCE.format(context=context, question=question)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject prior conversation as native chat turns
    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    usage = response.usage
    log_call(
        call_type="chat_completion",
        model=LLM_MODEL,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        metadata={"question": question[:200]},
    )

    return response.choices[0].message.content


def ask(
    question: str,
    openai_client: OpenAI | None = None,
    qdrant_client: QdrantClient | None = None,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Full RAG pipeline: classify → (rewrite) → retrieve → generate answer.

    Returns:
        {
            "answer": "...",
            "sources": [...],
            "chunks_used": int,
            "intent": "greeting" | "off_topic" | "policy_query" | "follow_up",
            "rewritten_query": str | None,
        }
    """
    oai = openai_client or get_openai_client()
    qd = qdrant_client or get_qdrant_client()

    # 0. Sanitize and truncate history
    history = sanitize_history(chat_history)
    history = truncate_history(history)

    # 1. Classify intent + rewrite if needed
    classification = classify_intent(question, history, openai_client=oai)
    intent = classification["intent"]

    # 2. For greeting/off_topic — return direct response, skip retrieval
    if intent in ("greeting", "off_topic"):
        return {
            "answer": classification["response"],
            "sources": [],
            "chunks_used": 0,
            "intent": intent,
            "rewritten_query": None,
        }

    # 3. Use rewritten query for retrieval (important for follow-ups)
    search_query = classification["rewritten_query"] or question
    raw_chunks = retrieve(search_query, oai, qd)

    # 4. Retrieval guardrails — filter by score, determine confidence tier
    chunks, confidence = check_retrieval(raw_chunks)

    # 5. Build context (empty string for "none" tier)
    context = build_context(chunks) if chunks else ""

    # 6. Generate answer — LLM is ALWAYS called, with tier-appropriate instructions
    answer = generate_answer(
        question, context, oai,
        chat_history=history,
        retrieval_confidence=confidence,
    )

    # 7. Extract sources (only from chunks that survived filtering)
    sources = build_sources(chunks)

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
        "intent": intent,
        "rewritten_query": classification["rewritten_query"],
        "retrieval_confidence": confidence,
    }


def _sse_event(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


def ask_stream(
    question: str,
    openai_client: OpenAI | None = None,
    qdrant_client: QdrantClient | None = None,
    chat_history: list[dict] | None = None,
):
    """
    Streaming RAG pipeline — yields SSE-formatted events.

    All intent types (greeting, off_topic, policy_query, follow_up) flow
    through the same SSE interface for frontend consistency.

    Event sequence:
      1. {"type": "intent", "intent": "...", "rewritten_query": "..."}
      2. {"type": "token", "content": "..."}  (one per token/word)
      3. {"type": "sources", "sources": [...]}
      4. {"type": "done"}
    """
    oai = openai_client or get_openai_client()
    qd = qdrant_client or get_qdrant_client()

    # 0. Sanitize and truncate history
    history = sanitize_history(chat_history)
    history = truncate_history(history)

    # 1. Classify intent + rewrite if needed
    classification = classify_intent(question, history, openai_client=oai)
    intent = classification["intent"]

    # Send intent event
    yield _sse_event({
        "type": "intent",
        "intent": intent,
        "rewritten_query": classification.get("rewritten_query"),
    })

    # 2. For greeting/off_topic — stream canned response word-by-word
    if intent in ("greeting", "off_topic"):
        words = classification["response"].split()
        for i, word in enumerate(words):
            # Add trailing space except for last word
            token = word if i == len(words) - 1 else word + " "
            yield _sse_event({"type": "token", "content": token})
        yield _sse_event({"type": "sources", "sources": []})
        yield _sse_event({"type": "done"})
        return

    # 3. Use rewritten query for retrieval
    search_query = classification["rewritten_query"] or question
    raw_chunks = retrieve(search_query, oai, qd)

    # 4. Retrieval guardrails
    chunks, confidence = check_retrieval(raw_chunks)

    # 5. Build context
    context = build_context(chunks) if chunks else ""

    # 6. Stream answer token-by-token from LLM
    for token in generate_answer_stream(
        question, context, oai,
        chat_history=history,
        retrieval_confidence=confidence,
    ):
        yield _sse_event({"type": "token", "content": token})

    # 7. Send sources
    sources = build_sources(chunks)
    yield _sse_event({"type": "sources", "sources": sources})

    # 8. Done
    yield _sse_event({"type": "done"})
