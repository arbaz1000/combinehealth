"""
Read parsed chunks from JSONL → embed with OpenAI → store in Qdrant.

Uses OpenAI text-embedding-3-small for dense vectors.
All API costs are tracked via app/cost_tracker.py.

Usage:
    python scripts/embed_and_index.py                # Index all chunks
    python scripts/embed_and_index.py --recreate     # Drop + recreate collection first
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from qdrant_client import QdrantClient, models

from app.config import (
    OPENAI_API_KEY,
    CHUNKS_DIR,
    QDRANT_PATH,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    INSURER,
)
from app.cost_tracker import log_call, print_summary


def load_chunks(insurer: str | None = None) -> list[dict]:
    insurer = insurer or INSURER
    """Load chunks from JSONL file."""
    chunks_path = CHUNKS_DIR / f"{insurer}_chunks.jsonl"
    chunks = []
    with open(chunks_path) as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    return chunks


def embed_texts(client: OpenAI, texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """
    Embed texts using OpenAI API in batches.
    Logs cost for each batch.
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        # Log cost
        usage = response.usage
        log_call(
            call_type="embedding",
            model=EMBEDDING_MODEL,
            input_tokens=usage.total_tokens,
            metadata={"batch_index": i // batch_size, "batch_size": len(batch)},
        )

        # Extract embeddings in order
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        done = min(i + batch_size, len(texts))
        print(f"  Embedded {done}/{len(texts)} chunks ({usage.total_tokens} tokens)")

    return all_embeddings


def build_payload(chunk: dict) -> dict:
    """Build Qdrant payload from chunk metadata."""
    meta = chunk["metadata"]
    return {
        "text": chunk["text"],
        "policy_name": meta.get("policy_name", ""),
        "policy_number": meta.get("policy_number", ""),
        "effective_date": meta.get("effective_date", ""),
        "source_url": meta.get("source_url", ""),
        "filename": meta.get("filename", ""),
        "insurer": meta.get("insurer", INSURER),
        "section_name": meta.get("section_name", ""),
        "chunk_index": meta.get("chunk_index", 0),
        "applicable_cpt_codes": meta.get("applicable_cpt_codes", []),
    }


def create_collection(client: QdrantClient, recreate: bool = False):
    """Create Qdrant collection for dense vectors."""
    exists = client.collection_exists(QDRANT_COLLECTION)

    if exists and recreate:
        print(f"Dropping existing collection: {QDRANT_COLLECTION}")
        client.delete_collection(QDRANT_COLLECTION)
        exists = False

    if not exists:
        print(f"Creating collection: {QDRANT_COLLECTION}")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Collection created ({EMBEDDING_DIM}-dim, cosine distance).")
    else:
        print(f"Collection {QDRANT_COLLECTION} already exists. Use --recreate to rebuild.")


def index_chunks(qdrant: QdrantClient, openai_client: OpenAI, chunks: list[dict], batch_size: int = 100):
    """Embed all chunks and upsert to Qdrant."""
    texts = [c["text"] for c in chunks]
    payloads = [build_payload(c) for c in chunks]

    print(f"\nEmbedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
    embeddings = embed_texts(openai_client, texts, batch_size=batch_size)

    print(f"\nUploading to Qdrant...")
    # Upsert in batches of 100
    upload_batch_size = 100
    for i in range(0, len(chunks), upload_batch_size):
        batch_end = min(i + upload_batch_size, len(chunks))
        points = [
            models.PointStruct(
                id=j,
                vector=embeddings[j],
                payload=payloads[j],
            )
            for j in range(i, batch_end)
        ]
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)

    # Verify
    info = qdrant.get_collection(QDRANT_COLLECTION)
    print(f"\nDone! Collection '{QDRANT_COLLECTION}' has {info.points_count} points.")


def main():
    parser = argparse.ArgumentParser(description="Embed chunks and index in Qdrant")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for OpenAI embedding API calls")
    args = parser.parse_args()

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    qdrant = QdrantClient(path=QDRANT_PATH)

    create_collection(qdrant, recreate=args.recreate)

    chunks = load_chunks()
    index_chunks(qdrant, openai_client, chunks, batch_size=args.batch_size)

    # Show cost summary
    print_summary()

    qdrant.close()


if __name__ == "__main__":
    main()