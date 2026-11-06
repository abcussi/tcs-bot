"""Gold embedders (AC-005.7): OpenAI ``text-embedding-3-small`` or an offline LSA embedder.

The LSA embedder is TF-IDF (unigrams) followed by a truncated SVD, fitted on the gold
corpus. Vectors are L2-normalized and zero-padded to ``dim`` so the Elasticsearch index dimension
is stable whatever the corpus size.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tcs_bot.logging_setup import get_logger
from tcs_bot.rag.embeddings import Embedder, HashEmbedder

if TYPE_CHECKING:
    from tcs_bot.config import Settings

log = get_logger(__name__)

_SLUG = re.compile(r"[^a-z0-9]+")


def model_slug(embedder_or_model: Embedder | str) -> str:
    """Collection-safe slug of an embedding model id (``lsa-256`` -> ``lsa_256``)."""
    model = embedder_or_model if isinstance(embedder_or_model, str) else embedder_or_model.model
    return _SLUG.sub("_", model.lower()).strip("_") or "unknown"


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)


def target_components(n_docs: int) -> int:
    """SVD rank that scales with the corpus (``max(6, 0.3 * n_docs)``).

    With ``n_docs - 1`` components on a small corpus the projection is nearly the raw TF-IDF space
    (pairwise cosines ~0.02, nothing clusters); a low rank keeps the shared topics.
    """
    return max(6, round(0.3 * n_docs))


class LsaEmbedder:
    """Offline TF-IDF + TruncatedSVD ("latent semantic analysis") embedder.

    Fit once on the corpus, then ``embed`` any text (queries included) into the same space.
    Deterministic: ``random_state=42``. Small corpora degrade gracefully:

    * ``n_components = min(dim, n_features - 1, n_docs - 1, target_components(n_docs))``; when
      that is < 2 the raw TF-IDF vector (vocabulary capped at ``dim``) is used instead of the SVD
      projection;
    * an empty vocabulary (e.g. only stop words) falls back to the offline hash embedder.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim < 2:
            raise ValueError("dim must be >= 2")
        self.dim = dim
        self.model = f"lsa-{dim}"
        self._vectorizer: Any = None
        self._svd: Any = None
        self._hash: HashEmbedder | None = None
        self.n_components = 0

    # ------------------------------------------------------------------ fit / embed
    @property
    def fitted(self) -> bool:
        return self._vectorizer is not None or self._hash is not None

    def _make_vectorizer(self, n_docs: int, max_features: int | None = None) -> Any:
        from sklearn.feature_extraction.text import TfidfVectorizer

        return TfidfVectorizer(
            sublinear_tf=True,
            # unigrams: bigrams are almost all singletons on a small corpus and only add noise
            ngram_range=(1, 1),
            stop_words="english",
            # words in over 60% of the documents carry no topic; (max_df < 1 would empty the
            # vocabulary of a 1-2 document corpus)
            max_df=0.6 if n_docs >= 5 else 1.0,
            min_df=1,
            max_features=max_features,
        )

    def fit(self, texts: list[str]) -> LsaEmbedder:
        from sklearn.decomposition import TruncatedSVD

        texts = [t or "" for t in texts]
        self._vectorizer, self._svd, self._hash = None, None, None
        n_docs = len(texts)
        if n_docs == 0:
            self._hash = HashEmbedder(self.dim)
            return self
        vectorizer = self._make_vectorizer(n_docs)
        try:
            tfidf = vectorizer.fit_transform(texts)
        except ValueError:  # empty vocabulary
            log.warning("gold.lsa.empty_vocabulary", n_docs=n_docs)
            self._hash = HashEmbedder(self.dim)
            return self
        n_features = tfidf.shape[1]
        n_comp = min(self.dim, n_features - 1, n_docs - 1, target_components(n_docs))
        if n_comp >= 2:
            self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
            self._svd.fit(tfidf)
            self._vectorizer = vectorizer
            self.n_components = n_comp
        else:
            # Too few docs for a meaningful SVD: use the TF-IDF space itself (<= dim terms).
            self._vectorizer = self._make_vectorizer(n_docs, max_features=self.dim)
            self._vectorizer.fit(texts)
            self.n_components = len(self._vectorizer.vocabulary_)
        log.info("gold.lsa.fitted", n_docs=n_docs, n_features=n_features, dim=self.n_components)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        """Float32 ``(len(texts), dim)``, L2-normalized (all-zero for out-of-vocabulary text)."""
        if not self.fitted:
            raise RuntimeError("LsaEmbedder.embed() called before fit()/load()")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._hash is not None:
            return self._hash.embed(texts)
        tfidf = self._vectorizer.transform([t or "" for t in texts])
        dense = self._svd.transform(tfidf) if self._svd is not None else tfidf.toarray()
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        out[:, : dense.shape[1]] = _l2_normalize(np.asarray(dense, dtype=np.float64))
        return out

    # ------------------------------------------------------------------ explainability
    def top_terms(self, text: str, n: int = 10) -> list[str]:
        """Highest-weighted TF-IDF terms of ``text`` in the fitted vocabulary."""
        if self._vectorizer is None:
            return []
        row = self._vectorizer.transform([text]).toarray()[0]
        idx = [i for i in np.argsort(-row)[:n] if row[i] > 0]
        names = self._vectorizer.get_feature_names_out()
        return [str(names[i]) for i in idx]

    def component_terms(self, component: int, n: int = 10) -> list[str]:
        """Terms with the largest loadings on one SVD component (empty without an SVD)."""
        if self._svd is None or not 0 <= component < self.n_components:
            return []
        names = self._vectorizer.get_feature_names_out()
        loadings = self._svd.components_[component]
        return [str(names[i]) for i in np.argsort(-loadings)[:n]]

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "dim": self.dim,
                "vectorizer": self._vectorizer,
                "svd": self._svd,
                "hash": self._hash is not None,
                "n_components": self.n_components,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> LsaEmbedder:
        """Load a model written by :meth:`save` (a trusted, locally generated artifact)."""
        import joblib

        state = joblib.load(Path(path))
        emb = cls(dim=int(state["dim"]))
        emb._vectorizer = state["vectorizer"]
        emb._svd = state["svd"]
        emb._hash = HashEmbedder(emb.dim) if state["hash"] else None
        emb.n_components = int(state["n_components"])
        return emb


def get_gold_embedder(settings: Settings, corpus: list[str]) -> Embedder:
    """OpenAI ``text-embedding-3-small`` when configured, else an LSA embedder fit on ``corpus``."""
    if not settings.use_fake_llm:
        from tcs_bot.rag.embeddings import OpenAIEmbedder

        return OpenAIEmbedder(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key.get_secret_value(),
        )
    return LsaEmbedder(dim=settings.lsa_dim).fit(corpus)
