"""
Centralized configuration — loads from environment variables and insurer YAML configs.

Simple mental model: This is the single place that knows all settings.
Every other module imports from here instead of reading env vars directly.
"""

import os
from pathlib import Path
from functools import lru_cache

import yaml
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
CONFIG_DIR = PROJECT_ROOT / "config" / "insurers"

# ── API Keys ───────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Embedding settings ─────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI, 1536-dim, $0.020/1M tokens
EMBEDDING_DIM = 1536

# ── Qdrant settings ───────────────────────────────────────────────────
QDRANT_PATH = str(DATA_DIR / "qdrant_store")  # local persistent storage

# ── LLM settings ──────────────────────────────────────────────────────
LLM_MODEL = "gpt-4o-mini"  # cheap + fast, good enough for RAG
LLM_TEMPERATURE = 0.1  # low temp = more factual, less creative
LLM_MAX_TOKENS = 1024

# ── RAG settings ──────────────────────────────────────────────────────
TOP_K = 8  # number of chunks to retrieve

# ── Retrieval guardrail settings ─────────────────────────────────────
# Chunks scoring below this are dropped before reaching the LLM.
RETRIEVAL_SCORE_THRESHOLD = 0.35
# If the best surviving chunk is below this, the LLM gets a low-confidence caveat.
RETRIEVAL_LOW_CONFIDENCE_THRESHOLD = 0.5

# ── Conversation settings ─────────────────────────────────────────────
# Max turn-pairs (user+assistant) to include in LLM context.
# 5 pairs = 10 messages. Increase for longer context at higher token cost.
# In production, consider token-budget-based truncation or summarization.
MAX_HISTORY_TURNS = 5


# ── Insurer configuration ─────────────────────────────────────────────
# Set INSURER env var to switch providers (e.g., INSURER=aetna).
# Each insurer needs a YAML config at config/insurers/{insurer}.yaml
# and a pre-built Qdrant collection.

@lru_cache()
def load_insurer_config(insurer: str | None = None) -> dict:
    """Load insurer-specific config from YAML."""
    insurer = insurer or INSURER
    config_path = CONFIG_DIR / f"{insurer}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


INSURER = os.getenv("INSURER", "uhc")
_insurer_cfg = load_insurer_config(INSURER)

# Provider display names — used across prompts, UI, and guardrails
INSURER_NAME = _insurer_cfg["name"]                    # "UnitedHealthcare"
INSURER_SHORT_NAME = _insurer_cfg["short_name"]        # "UHC"
INSURER_CONTACT = _insurer_cfg.get("contact_info", f"{INSURER_SHORT_NAME} directly")
INSURER_PORTAL = _insurer_cfg.get("provider_portal", f"the official {INSURER_SHORT_NAME} provider portal")
QDRANT_COLLECTION = _insurer_cfg["collection_name"]    # "uhc_policies"