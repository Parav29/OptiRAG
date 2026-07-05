"""Hybrid RAG retriever: BM25 top-20 union dense top-20 -> Reciprocal Rank
Fusion -> CrossEncoder rerank -> top 5.

The ``rag_enabled`` flag is the ablation switch: when False the retriever
returns an EMPTY RetrievedContext and the downstream Modeling Agent gets no
grounding context. This is what the eval harness toggles to produce the
RAG-on / RAG-off comparison.
"""

import logging

from src import config
from src.kb_index import KBIndex, get_index
from src.schemas import ProblemIntake, RetrievedContext

logger = logging.getLogger(__name__)

_reranker = None
_reranker_failed = False


def _get_reranker():
    global _reranker, _reranker_failed
    if _reranker is None and not _reranker_failed:
        try:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(config.RERANKER_MODEL)
        except Exception as exc:  # pragma: no cover - environment dependent
            _reranker_failed = True
            logger.warning(
                "CrossEncoder unavailable (%s); using RRF order without rerank.", exc
            )
    return _reranker


def _rrf(rankings: list[list[int]], k: int) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


def retrieve(
    intake: ProblemIntake,
    rag_enabled: bool = True,
    index: KBIndex | None = None,
) -> RetrievedContext:
    if not rag_enabled:
        return RetrievedContext()

    index = index or get_index()
    query = f"{intake.problem_family} problem formulation: {intake.raw_text}"

    bm25_hits = index.bm25_search(query, config.RETRIEVAL_CANDIDATES)
    dense_hits = index.dense_search(query, config.RETRIEVAL_CANDIDATES)

    rankings = [[i for i, _ in bm25_hits]]
    if dense_hits:
        rankings.append([i for i, _ in dense_hits])
    fused = _rrf(rankings, config.RRF_K)
    candidates = sorted(fused, key=fused.get, reverse=True)

    reranker = _get_reranker()
    if reranker is not None and candidates:
        pairs = [(query, index.chunks[i].text) for i in candidates]
        scores = reranker.predict(pairs)
        order = sorted(range(len(candidates)), key=lambda j: scores[j], reverse=True)
        top = [(candidates[j], float(scores[j])) for j in order[: config.RETRIEVAL_TOP_K]]
    else:
        top = [(i, fused[i]) for i in candidates[: config.RETRIEVAL_TOP_K]]

    return RetrievedContext(
        chunks=[index.chunks[i].text for i, _ in top],
        scores=[s for _, s in top],
        source_ids=[index.chunks[i].source_id for i, _ in top],
    )
