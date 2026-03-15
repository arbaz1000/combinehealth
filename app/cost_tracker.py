"""
Track costs of all OpenAI API calls (embeddings + LLM).

Two output files:
  data/costs/call_log.jsonl  — every API call with timestamp, model, tokens, cost
  data/costs/summary.json    — running totals by call type (updated after each call)

Usage:
    from app.cost_tracker import tracked_openai_client
    client = tracked_openai_client()
    # Use client.chat.completions.create() / client.embeddings.create() as normal
    # Costs are logged automatically.

    # To view costs:
    from app.cost_tracker import get_summary, get_recent_calls
    print(get_summary())
    print(get_recent_calls(10))

Pricing as of March 2026 (update if models change):
    text-embedding-3-small: $0.020 / 1M tokens
    gpt-4o-mini:            $0.150 / 1M input tokens, $0.600 / 1M output tokens
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.config import PROJECT_ROOT

# ── Paths ──────────────────────────────────────────────────────────────
COSTS_DIR = PROJECT_ROOT / "data" / "costs"
CALL_LOG_PATH = COSTS_DIR / "call_log.jsonl"
SUMMARY_PATH = COSTS_DIR / "summary.json"

# ── Pricing (USD per 1M tokens) ───────────────────────────────────────
PRICING = {
    "text-embedding-3-small": {"input": 0.020, "output": 0.0},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    # Add more models here as needed
}

# Thread lock for file writes (safe for concurrent API calls)
_lock = threading.Lock()


def _ensure_dirs():
    COSTS_DIR.mkdir(parents=True, exist_ok=True)


def _calculate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Calculate USD cost for a given model and token counts."""
    prices = PRICING.get(model, {"input": 0.0, "output": 0.0})
    cost = (input_tokens * prices["input"] / 1_000_000) + \
           (output_tokens * prices["output"] / 1_000_000)
    return round(cost, 8)


def log_call(
    call_type: str,
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
    metadata: dict | None = None,
):
    """
    Log a single API call and update the running summary.

    Args:
        call_type: "embedding" or "chat_completion"
        model: model name (e.g., "gpt-4o-mini")
        input_tokens: prompt/input token count
        output_tokens: completion/output token count (0 for embeddings)
        metadata: optional extra info (e.g., question text, chunk count)
    """
    _ensure_dirs()
    cost = _calculate_cost(model, input_tokens, output_tokens)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "call_type": call_type,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": cost,
    }
    if metadata:
        entry["metadata"] = metadata

    with _lock:
        # Append to call log
        with open(CALL_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Update summary
        _update_summary(call_type, model, input_tokens, output_tokens, cost)


def _update_summary(call_type: str, model: str, input_tokens: int, output_tokens: int, cost: float):
    """Update the running totals in summary.json."""
    summary = _load_summary()

    key = f"{call_type}:{model}"
    if key not in summary["by_type"]:
        summary["by_type"][key] = {
            "call_count": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
        }

    entry = summary["by_type"][key]
    entry["call_count"] += 1
    entry["total_input_tokens"] += input_tokens
    entry["total_output_tokens"] += output_tokens
    entry["total_cost_usd"] = round(entry["total_cost_usd"] + cost, 8)

    summary["total_cost_usd"] = round(summary["total_cost_usd"] + cost, 8)
    summary["total_calls"] += 1
    summary["last_updated"] = datetime.now(timezone.utc).isoformat()

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)


def _load_summary() -> dict:
    """Load existing summary or create a fresh one."""
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH) as f:
            return json.load(f)
    return {
        "total_cost_usd": 0.0,
        "total_calls": 0,
        "last_updated": None,
        "by_type": {},
    }


def get_summary() -> dict:
    """Get the current cost summary. Call this to check spending."""
    _ensure_dirs()
    return _load_summary()


def get_recent_calls(n: int = 10) -> list[dict]:
    """Get the N most recent API calls from the log."""
    _ensure_dirs()
    if not CALL_LOG_PATH.exists():
        return []
    calls = []
    with open(CALL_LOG_PATH) as f:
        for line in f:
            if line.strip():
                calls.append(json.loads(line))
    return calls[-n:]


def print_summary():
    """Pretty-print the cost summary to stdout."""
    s = get_summary()
    print(f"\n{'='*50}")
    print(f"  OpenAI Cost Summary")
    print(f"  Total: ${s['total_cost_usd']:.6f} ({s['total_calls']} calls)")
    print(f"  Last updated: {s.get('last_updated', 'never')}")
    print(f"{'='*50}")
    for key, data in s.get("by_type", {}).items():
        print(f"  {key}:")
        print(f"    Calls: {data['call_count']}")
        print(f"    Tokens: {data['total_input_tokens']:,} in / {data['total_output_tokens']:,} out")
        print(f"    Cost: ${data['total_cost_usd']:.6f}")
    print()
