"""Central configuration. Everything overridable via environment / .env."""

import os

from dotenv import load_dotenv

load_dotenv()

# LLM provider: Google Gemini (via the google-genai SDK).
# GEMINI_API_KEY is preferred; GOOGLE_API_KEY is accepted as a fallback so the
# SDK's own default env var also works.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", GEMINI_MODEL)
# Generous default: Gemini 2.5 models spend part of the output budget on
# internal "thinking", so a low cap can truncate structured JSON.
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))

# Retrieval
QDRANT_URL = os.getenv("QDRANT_URL", "")  # empty -> in-memory Qdrant (tests/dev)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "optiagent_kb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
KB_DIR = os.getenv("KB_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "kb"))

RETRIEVAL_CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "20"))  # per retriever
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))  # after rerank
RRF_K = int(os.getenv("RRF_K", "60"))

# Pipeline
MAX_MODELING_ATTEMPTS = int(os.getenv("MAX_MODELING_ATTEMPTS", "3"))
