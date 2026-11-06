"""Numpy cosine-similarity retrieval over a vendor's chunk matrix (AC-003.4)."""

from __future__ import annotations

import numpy as np

from vendor_risk_ai.rag.embeddings import Embedder
from vendor_risk_ai.schemas.domain import ChunkRecord


def _l2_normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)


def cosine_top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Indices + cosine scores of the ``k`` rows of ``matrix`` most similar to ``query``.

    Sorted by score desc (ties by index asc). Zero vectors score 0.0; ``k > n`` returns ``n`` rows.
    """
    q = np.asarray(query, dtype=np.float64).ravel()
    m = np.asarray(matrix, dtype=np.float64)
    if k <= 0 or m.size == 0:
        return []
    if m.ndim == 1:
        m = m.reshape(1, -1)
    if m.shape[1] != q.shape[0]:
        raise ValueError(f"dimension mismatch: query {q.shape[0]} vs matrix {m.shape[1]}")

    q_norm = float(np.linalg.norm(q))
    q = q / q_norm if q_norm > 0 else q
    scores = np.clip(_l2_normalize_rows(m) @ q, -1.0, 1.0)

    n = scores.shape[0]
    k = min(k, n)
    candidates = np.arange(n) if k == n else np.argpartition(-scores, k - 1)[:k]
    # lexsort: last key is primary -> score desc, then index asc for determinism
    order = candidates[np.lexsort((candidates, -scores[candidates]))]
    return [(int(i), float(scores[i])) for i in order]


class Retriever:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def _usable(self, chunk: ChunkRecord) -> bool:
        emb = chunk.embedding
        if not emb or len(emb) != self.embedder.dim:
            return False
        # Same dim but a different model (e.g. OpenAI vs hash) lives in another vector space.
        return chunk.embedding_model in (None, self.embedder.model)

    def chunk_matrix(self, chunks: list[ChunkRecord]) -> np.ndarray:
        """Stack stored embeddings, embedding missing/incompatible ones on the fly."""
        matrix = np.zeros((len(chunks), self.embedder.dim), dtype=np.float32)
        missing: list[int] = []
        for i, c in enumerate(chunks):
            if self._usable(c):
                matrix[i] = np.asarray(c.embedding, dtype=np.float32)
            else:
                missing.append(i)
        if missing:
            matrix[missing] = self.embedder.embed([chunks[i].text for i in missing])
        return matrix

    def retrieve(
        self, query: str, chunks: list[ChunkRecord], k: int
    ) -> list[tuple[ChunkRecord, float]]:
        if not chunks or k <= 0:
            return []
        q = self.embedder.embed([query])[0]
        hits = cosine_top_k(q, self.chunk_matrix(chunks), k)
        return [(chunks[i], score) for i, score in hits]
