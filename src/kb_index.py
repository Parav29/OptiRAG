"""Knowledge-base indexing: markdown -> chunks -> Qdrant (dense) + BM25 (sparse).

Chunking is heading-based: each ``##`` section of each KB markdown file is one
chunk (the docs are written so a section is roughly 200-300 tokens). Dense
vectors live in Qdrant — a real server when QDRANT_URL is set (docker-compose)
or an in-memory instance otherwise (tests / local dev without docker). If
sentence-transformers cannot be loaded (offline environment), the index
degrades to BM25-only and says so loudly.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from src import config

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    source_id: str  # e.g. "diet.md#2-decision-variables"
    text: str


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def load_chunks(kb_dir: str | Path | None = None) -> list[Chunk]:
    kb_dir = Path(kb_dir or config.KB_DIR)
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        sections = re.split(r"(?m)^## ", path.read_text())
        for i, section in enumerate(sections):
            body = section.strip()
            if not body:
                continue
            title = body.splitlines()[0].strip().lstrip("# ")
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]
            text = body if i == 0 else f"## {body}"
            chunks.append(Chunk(source_id=f"{path.name}#{i}-{slug}", text=text))
    return chunks


class KBIndex:
    """Hybrid index over the KB chunks: BM25 always, dense when available."""

    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        if not self.chunks:
            raise RuntimeError(f"No KB chunks found in {config.KB_DIR}.")
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])
        self._embedder = None
        self._qdrant = None
        self._dense_ready = False
        self._build_dense()

    # -- sparse ---------------------------------------------------------
    def bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in ranked[:top_k]]

    # -- dense ----------------------------------------------------------
    def _build_dense(self) -> None:
        try:
            from qdrant_client import QdrantClient, models
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(config.EMBEDDING_MODEL)
            self._qdrant = (
                QdrantClient(url=config.QDRANT_URL)
                if config.QDRANT_URL
                else QdrantClient(location=":memory:")
            )
            vectors = self._embedder.encode(
                [c.text for c in self.chunks], normalize_embeddings=True
            )
            self._qdrant.recreate_collection(
                collection_name=config.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=vectors.shape[1], distance=models.Distance.COSINE
                ),
            )
            self._qdrant.upsert(
                collection_name=config.QDRANT_COLLECTION,
                points=[
                    models.PointStruct(
                        id=i, vector=vectors[i].tolist(), payload={"idx": i}
                    )
                    for i in range(len(self.chunks))
                ],
            )
            self._dense_ready = True
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning(
                "Dense index unavailable (%s: %s); retrieval degrades to BM25-only.",
                type(exc).__name__,
                exc,
            )

    def dense_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if not self._dense_ready:
            return []
        vector = self._embedder.encode([query], normalize_embeddings=True)[0]
        hits = self._qdrant.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=vector.tolist(),
            limit=top_k,
        ).points
        return [(hit.payload["idx"], float(hit.score)) for hit in hits]

    @property
    def dense_ready(self) -> bool:
        return self._dense_ready


_default_index: KBIndex | None = None


def get_index() -> KBIndex:
    global _default_index
    if _default_index is None:
        _default_index = KBIndex()
    return _default_index
