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
QDRANT_COLLECTION = "uhc_policies"

# ── LLM settings ──────────────────────────────────────────────────────
LLM_MODEL = "gpt-4o-mini"  # cheap + fast, good enough for RAG
LLM_TEMPERATURE = 0.1  # low temp = more factual, less creative
LLM_MAX_TOKENS = 1024

# ── RAG settings ──────────────────────────────────────────────────────
TOP_K = 8  # number of chunks to retrieve

# ── Conversation settings ─────────────────────────────────────────────
# Max turn-pairs (user+assistant) to include in LLM context.
# 5 pairs = 10 messages. Increase for longer context at higher token cost.
# In production, consider token-budget-based truncation or summarization.
MAX_HISTORY_TURNS = 5


@lru_cache()
def load_insurer_config(insurer: str = "uhc") -> dict:
    """Load insurer-specific config from YAML."""
    config_path = CONFIG_DIR / f"{insurer}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
