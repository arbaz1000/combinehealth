"""
RAG pipeline: retrieve relevant policy chunks from Qdrant, build prompt, call LLM.

Simple mental model:
1. User asks a question
2. We embed the question with OpenAI, search Qdrant for similar chunks
3. We stuff the top results into a prompt with instructions
4. GPT-4o-mini reads the context and answers

All OpenAI calls (embedding + LLM) are cost-tracked automatically.
"""

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
)
from app.cost_tracker import log_call

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

USER_PROMPT_TEMPLATE = """Based on the following UHC policy excerpts, answer the user's question.

--- POLICY CONTEXT ---
{context}
--- END CONTEXT ---

User question: {question}

Provide a clear, structured answer with policy citations."""


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


def generate_answer(question: str, context: str, client: OpenAI) -> str:
    """Call GPT-4o-mini with the RAG context. Cost-tracked."""
    user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
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


def ask(question: str, openai_client: OpenAI | None = None, qdrant_client: QdrantClient | None = None) -> dict:
    """
    Full RAG pipeline: embed query → retrieve → build context → generate answer.

    Returns:
        {
            "answer": "...",
            "sources": [{"policy_name": ..., "policy_number": ..., "source_url": ...}],
            "chunks_used": int,
        }
    """
    oai = openai_client or get_openai_client()
    qd = qdrant_client or get_qdrant_client()

    # 1. Retrieve relevant chunks
    chunks = retrieve(question, oai, qd)

    if not chunks:
        return {
            "answer": "I couldn't find any relevant policy information for your question. Please try rephrasing or ask about a specific procedure, diagnosis, or CPT code.",
            "sources": [],
            "chunks_used": 0,
        }

    # 2. Build context from retrieved chunks
    context = build_context(chunks)

    # 3. Generate answer
    answer = generate_answer(question, context, oai)

    # 4. Extract sources
    sources = build_sources(chunks)

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
    }
