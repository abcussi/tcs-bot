"""Near-duplicate collapsing for gold (AC-005.8).

The same story is often crawled twice: a search-result snippet at depth 0 and the full page at
depth 1 (Bing News), or a Hacker News hit and its ``items/{id}`` thread with an identical title.
Both inflate hub centrality and form tiny clusters, so gold merges them *at build time*: silver is
never modified. The richest (longest) document is kept and the others are recorded on it.

Two documents are duplicates when ``normalised title`` is equal (and the texts are not
unrelated), or their texts are near-identical (high TF-IDF cosine), or titles overlap a lot and the
texts are close. Groups are the connected components of that relation.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from tcs_bot.medallion.models import SilverRecord

# Same normalised title: still require this much text overlap (monthly "Who is hiring?" threads
# share a title but not content).
SAME_TITLE_MIN_COSINE = 0.1
# Near-identical text, whatever the titles say.
TEXT_DUP_COSINE = 0.9
# Strongly overlapping titles plus close texts (outlet-specific headline edits).
TITLE_JACCARD_MIN = 0.7
TITLE_JACCARD_COSINE = 0.6
MIN_TITLE_TOKENS = 3

_SITE_SUFFIX = re.compile("\\s+[-|\\u2013\\u2014]\\s+[^-|\\u2013\\u2014]{2,40}$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
# Broken encodings seen in the wild: U+FFFD and literal backslash-u escapes (apostrophes).
_ESCAPES = re.compile("[\\\\]u[0-9a-fA-F]{4}|[\\ufffd]")
# Same title but published this many days apart is a different post (recurring threads).
MAX_DAYS_APART = 7


def normalize_title(title: str) -> str:
    """Lower-case, accent-folded, punctuation-free title with a trailing ``- Outlet`` removed."""
    t = unicodedata.normalize("NFKD", _ESCAPES.sub(" ", title or ""))
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    stripped = _SITE_SUFFIX.sub("", t)
    if len(_NON_ALNUM.sub(" ", stripped.lower()).split()) >= MIN_TITLE_TOKENS:
        t = stripped
    return _NON_ALNUM.sub(" ", t.lower()).strip()


def _title_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def text_cosine(texts: Sequence[str]) -> np.ndarray:
    """Dense ``(n, n)`` TF-IDF cosine over the documents (zeros if the vocabulary is empty)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    n = len(texts)
    try:
        M = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=1).fit_transform(
            [t or "" for t in texts]
        )
    except ValueError:
        return np.eye(n)
    S = (M @ M.T).toarray()
    np.fill_diagonal(S, 1.0)
    return np.clip(S, 0.0, 1.0)


def is_duplicate(
    title_a: str,
    title_b: str,
    cosine: float,
    *,
    norm_a: str | None = None,
    norm_b: str | None = None,
) -> bool:
    """Pairwise rule (see module docstring)."""
    na = norm_a if norm_a is not None else normalize_title(title_a)
    nb = norm_b if norm_b is not None else normalize_title(title_b)
    if na and na == nb and cosine >= SAME_TITLE_MIN_COSINE:
        return True
    if cosine >= TEXT_DUP_COSINE:
        return True
    n_tok = min(len(na.split()), len(nb.split()))
    return (
        n_tok >= MIN_TITLE_TOKENS
        and _title_jaccard(na, nb) >= TITLE_JACCARD_MIN
        and cosine >= TITLE_JACCARD_COSINE
    )


def _days_apart(a: SilverRecord, b: SilverRecord) -> float:
    if a.published_at is None or b.published_at is None:
        return 0.0
    return abs((a.published_at - b.published_at).total_seconds()) / 86400.0


@dataclass
class DedupeResult:
    """Output of :func:`dedupe_records`."""

    records: list[SilverRecord]  # kept (richest) documents, original order
    merged: dict[int, list[SilverRecord]] = field(default_factory=dict)  # kept id -> dropped docs

    @property
    def n_removed(self) -> int:
        return sum(len(v) for v in self.merged.values())


def _richness(r: SilverRecord) -> tuple[int, int, float, int]:
    # longest text first, then deeper (full page), then riskier, then lowest id (deterministic)
    return (max(r.word_count, len(r.clean_text.split())), r.depth, r.risk_score, -r.doc_id)


def dedupe_records(
    records: Sequence[SilverRecord],
    text_of: Callable[[SilverRecord], str] | None = None,
    S: np.ndarray | None = None,
) -> DedupeResult:
    """Collapse near-duplicate documents, keeping the richest of each group.

    ``text_of`` (default ``title + clean_text``) feeds the TF-IDF cosine unless a precomputed
    similarity matrix ``S`` is passed. Order of the kept records follows the input order.
    """
    recs = list(records)
    n = len(recs)
    if n < 2:
        return DedupeResult(recs)
    if S is None:
        get = text_of or (lambda r: f"{r.title}\n{r.clean_text}")
        S = text_cosine([get(r) for r in recs])
    norm = [normalize_title(r.title) for r in recs]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            cos = float(S[i, j])
            if is_duplicate(recs[i].title, recs[j].title, cos, norm_a=norm[i], norm_b=norm[j]):
                if cos < TEXT_DUP_COSINE and _days_apart(recs[i], recs[j]) > MAX_DAYS_APART:
                    continue
                parent[find(j)] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    keep_idx: list[int] = []
    merged: dict[int, list[SilverRecord]] = {}
    for members in groups.values():
        best = max(members, key=lambda i: _richness(recs[i]))
        keep_idx.append(best)
        others = sorted((recs[i] for i in members if i != best), key=lambda r: r.doc_id)
        if others:
            merged[recs[best].doc_id] = others
    keep_idx.sort()
    return DedupeResult([recs[i] for i in keep_idx], merged)
