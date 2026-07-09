"""Central configuration. Everything overridable via environment / .env."""

import os

from dotenv import load_dotenv

load_dotenv()


def _clean_key(raw: str) -> str:
    """Strip whitespace and any surrounding quotes or angle brackets that get
    pasted in by accident (e.g. a settings field that wrapped the value in
    <...>), which would otherwise make the key fail auth."""
    return raw.strip().strip("<>").strip("\"'").strip()


# LLM provider: Google Gemini, used through its OpenAI-compatible REST
# endpoint so the whole `openai` SDK path in src/llm.py works unchanged. The
# provider lives entirely behind src/llm.py; swapping to another
# OpenAI-compatible endpoint is just a base URL + key change here.
LLM_API_KEY = _clean_key(
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("LLM_API_KEY", "")
)
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
# Small, fast model for the pipeline; a stronger one for the judge, since a
# grader should be at least as capable as what it grades. gemini-2.5-flash has
# a more workable free-tier daily quota than -flash-lite (which is capped at
# ~20 requests/day), so it is the default for both roles here.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-flash")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
# Gemini 2.5 models "think" by default, spending part of the max_tokens budget
# on hidden reasoning — which can leave the visible JSON answer empty. "none"
# turns thinking off so the whole budget goes to the answer and we don't burn
# extra tokens/quota. Set to "low"/"medium"/"high" to re-enable it.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "none")
# Hard per-request timeout (seconds) so a stalled endpoint can't hang the whole
# pipeline for the SDK's 600s default; the retry loop then takes over.
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "60"))

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

# Rate limiting — stay under the Gemini free-tier per-minute request limit
# (gemini-2.5-flash allows ~10-15 RPM on the free tier). Default 15/min; the
# client also backs off on 429s and honours server-suggested retry delays.
# Note the tighter bind is the per-DAY quota, not per-minute.
LLM_CALLS_PER_MIN = float(os.getenv("LLM_CALLS_PER_MIN", "15"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
LLM_MAX_BACKOFF_S = float(os.getenv("LLM_MAX_BACKOFF_S", "60"))
