"""Character-based chunking with sentence/word boundary preference (AC-003.2)."""

from __future__ import annotations

import re

# End of a sentence or paragraph: punctuation followed by whitespace, or a newline.
_SENTENCE_END = re.compile(r"[.!?;:](?=\s)|\n")


def _boundary(text: str, lo: int, hi: int) -> int | None:
    """Best cut position in ``text[lo:hi]``: last sentence end, else last whitespace."""
    window = text[lo:hi]
    last = None
    for m in _SENTENCE_END.finditer(window):
        last = m.end()
    if last is not None:
        return lo + last
    ws = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
    return lo + ws + 1 if ws >= 0 else None


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into chunks of at most ``size`` chars overlapping by ~``overlap`` chars.

    Cuts prefer sentence ends, then word boundaries; a word longer than the window is hard-cut.
    """
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    text = text.strip()
    if not text:
        return []

    n = len(text)
    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Search only the back half of the window (and past the overlap) so progress is
            # guaranteed and chunks don't get tiny.
            lo = start + max(size // 2, overlap + 1)
            if lo < end:
                cut = _boundary(text, lo, end)
                if cut is not None and cut > start:
                    end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        # Don't start mid-word: move to the next word start if one exists before ``end``.
        if 0 < next_start < end and not text[next_start - 1].isspace():
            ws = re.search(r"\s", text[next_start:end])
            if ws is not None:
                next_start += ws.end()
        start = next_start
    return chunks
