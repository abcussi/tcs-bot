"""Embeddings + numpy cosine retrieval (AC-003.3, AC-003.4)."""

from __future__ import annotations

import numpy as np
import pytest

from vendor_risk_ai.config import Settings
from vendor_risk_ai.rag.embeddings import Embedder, HashEmbedder, OpenAIEmbedder, get_embedder
from vendor_risk_ai.rag.retriever import Retriever, cosine_top_k
from vendor_risk_ai.schemas.domain import ChunkRecord


# ----------------------------------------------------------------------------- cosine_top_k
def test_cosine_top_k_matches_manual_calculation() -> None:
    """AC-003.4: scores equal manual cosine; sorted desc; top-k only."""
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(50, 16)).astype(np.float32)
    query = rng.normal(size=16).astype(np.float32)
    manual = [
        float(np.dot(row, query) / (np.linalg.norm(row) * np.linalg.norm(query))) for row in matrix
    ]
    expected = sorted(range(50), key=lambda i: -manual[i])[:5]

    got = cosine_top_k(query, matrix, 5)
    assert [i for i, _ in got] == expected
    for i, score in got:
        assert score == pytest.approx(manual[i], abs=1e-5)
    assert [s for _, s in got] == sorted((s for _, s in got), reverse=True)


def test_cosine_top_k_edge_cases() -> None:
    """AC-003.4: k > n, k <= 0, empty matrix, zero vectors, dim mismatch."""
    m = np.array([[1.0, 0.0], [0.0, 0.0], [0.5, 0.5]])
    got = cosine_top_k(np.array([1.0, 0.0]), m, 10)
    assert [i for i, _ in got] == [0, 2, 1]
    assert got[0][1] == pytest.approx(1.0)
    assert got[1][1] == pytest.approx(np.sqrt(0.5))
    assert got[2][1] == 0.0  # zero row -> 0, not NaN
    assert cosine_top_k(np.array([1.0, 0.0]), m, 0) == []
    assert cosine_top_k(np.array([1.0, 0.0]), np.zeros((0, 2)), 3) == []
    zero_q = cosine_top_k(np.zeros(2), m, 2)
    assert zero_q == [(0, 0.0), (1, 0.0)]  # ties broken by index
    with pytest.raises(ValueError):
        cosine_top_k(np.ones(3), m, 1)


def test_cosine_scores_are_clipped_to_unit_interval() -> None:
    """AC-003.4: floating error never yields |score| > 1 (SourceRef bound)."""
    v = np.full(1536, 1e-3, dtype=np.float32)
    ((_, score),) = cosine_top_k(v, v[None, :], 1)
    assert -1.0 <= score <= 1.0


# ----------------------------------------------------------------------------- embedders
def test_hash_embedder_shape_determinism_and_similarity() -> None:
    """AC-003.3: offline embedder is deterministic, dim 1536, L2-normalized, overlap-aware."""
    e = HashEmbedder()
    assert isinstance(e, Embedder)
    texts = [
        "auditor expressed substantial doubt about going concern",
        "going concern doubt raised by the auditor",
        "the cafeteria menu features pasta on fridays",
        "",
    ]
    v = e.embed(texts)
    assert v.shape == (4, 1536) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v[:3], axis=1), 1.0, atol=1e-5)
    assert not v[3].any()
    assert np.array_equal(v, HashEmbedder().embed(texts))
    assert float(v[0] @ v[1]) > float(v[0] @ v[2]) + 0.2
    assert e.embed([]).shape == (0, 1536)


def test_get_embedder_picks_by_provider() -> None:
    """AC-003.3: fake provider -> HashEmbedder; OpenAI -> text-embedding-3-small (no call made)."""
    assert isinstance(get_embedder(Settings(llm_provider="fake", _env_file=None)), HashEmbedder)
    real = get_embedder(
        Settings(llm_provider="openai", openai_api_key="sk-test-not-real", _env_file=None)
    )
    assert isinstance(real, OpenAIEmbedder)
    assert real.model == "text-embedding-3-small" and real.dim == 1536
    assert real._client.model == "text-embedding-3-small"


def test_openai_embedder_converts_to_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC-003.3: OpenAIEmbedder returns float32 (n, dim) from the LangChain client."""
    e = OpenAIEmbedder(api_key="sk-test-not-real", dim=3)
    monkeypatch.setattr(
        type(e._client), "embed_documents", lambda self, texts: [[1, 2, 3] for _ in texts]
    )
    out = e.embed(["a", "b"])
    assert out.shape == (2, 3) and out.dtype == np.float32


# ----------------------------------------------------------------------------- retriever
def test_retriever_uses_stored_embeddings_and_embeds_missing() -> None:
    """AC-003.4: stored embeddings reused when dim/model match; others embedded on the fly."""
    e = HashEmbedder()
    texts = [
        "SOC 2 report noted control exceptions in change management",
        "quarterly revenue grew with strong cash position",
        "ransomware attack caused a data breach at the vendor",
    ]
    stored = e.embed(texts[:1])[0].tolist()
    chunks = [
        ChunkRecord(
            document_id=1, chunk_index=0, text=texts[0], embedding=stored, embedding_model=e.model
        ),
        ChunkRecord(document_id=1, chunk_index=1, text=texts[1]),  # missing
        ChunkRecord(document_id=2, chunk_index=0, text=texts[2], embedding=[0.1, 0.2]),  # bad dim
    ]
    calls: list[list[str]] = []
    orig = e.embed

    def spy(batch: list[str]) -> np.ndarray:
        calls.append(batch)
        return orig(batch)

    e.embed = spy  # type: ignore[method-assign]
    hits = Retriever(e).retrieve("ransomware data breach", chunks, k=2)
    assert [h[0].text for h in hits][0] == texts[2]
    assert len(hits) == 2 and hits[0][1] >= hits[1][1]
    assert calls[1] == [texts[1], texts[2]]  # [query], then only the unusable chunks
    assert Retriever(e).retrieve("x", [], 3) == []


def test_retriever_ignores_embeddings_from_other_model() -> None:
    """AC-003.4: same-dim vectors from another model are re-embedded, not mixed."""
    e = HashEmbedder()
    foreign = ChunkRecord(
        chunk_index=0,
        text="going concern",
        embedding=[0.0] * 1536,
        embedding_model="text-embedding-3-small",
    )
    ((chunk, score),) = Retriever(e).retrieve("going concern", [foreign], 1)
    assert score == pytest.approx(1.0, abs=1e-5)
