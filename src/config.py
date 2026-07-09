"""Central configuration. Everything overridable via environment / .env."""

import os

from dotenv import load_dotenv

load_dotenv()


def _clean_key(raw: str) -> str:
    """Strip whitespace and any surrounding quotes or angle brackets that get
    pasted in by accident (e.g. a settings field that wrapped the value in
    <...>), which would otherwise make the key fail auth."""
    return raw.strip().strip("<>").strip("\"'").strip()


# LLM provider: NVIDIA NIM (build.nvidia.com), used through its
# OpenAI-compatible REST endpoint. The provider lives entirely behind
# src/llm.py; swapping to another OpenAI-compatible endpoint is just a base
# URL + key change here.
LLM_API_KEY = _clean_key(
    os.getenv("NVIDIA_API_KEY") or os.getenv("LLM_API_KEY", "")
)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
# Small, fast instruct model for the pipeline; a larger one for the judge,
# since a grader should be at least as capable as what it grades.
LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.1-8b-instruct")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "meta/llama-3.1-70b-instruct")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

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

# Rate limiting — stay under the NVIDIA NIM free-tier limit (40 requests/min
# per account). Default 30/min leaves headroom; the client also backs off on
# 429s.
LLM_CALLS_PER_MIN = float(os.getenv("LLM_CALLS_PER_MIN", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
LLM_MAX_BACKOFF_S = float(os.getenv("LLM_MAX_BACKOFF_S", "60"))
