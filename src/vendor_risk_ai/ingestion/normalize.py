"""Unicode / whitespace normalization (AC-003.2)."""

from __future__ import annotations

import re
import unicodedata

_HYPHEN_BREAK = re.compile(r"(\w)-[ \t]*\n[ \t]*(\w)")
_INLINE_WS = re.compile(r"[^\S\n]+")  # whitespace except newline
_MANY_NEWLINES = re.compile(r"\n{3,}")
_KEEP_CONTROL = {"\n", "\t"}


def _strip_control(s: str) -> str:
    # Cc = control chars, Cf = format chars (zero-width space, BOM, soft hyphen, ...)
    return "".join(
        ch for ch in s if ch in _KEEP_CONTROL or unicodedata.category(ch) not in ("Cc", "Cf")
    )


def normalize_text(s: str) -> str:
    """NFKC, strip control chars, join hyphenated line breaks, collapse whitespace.

    Paragraph breaks (blank lines) are kept as a single ``\\n\\n`` so the chunker can prefer them.
    """
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = unicodedata.normalize("NFKC", s)
    s = _strip_control(s)
    s = _HYPHEN_BREAK.sub(r"\1\2", s)
    s = _INLINE_WS.sub(" ", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    s = _MANY_NEWLINES.sub("\n\n", s)
    return s.strip()
