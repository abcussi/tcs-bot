"""Ingestion: extraction, normalization, chunking (AC-003.2)."""

from __future__ import annotations

import io

import pytest

from vendor_risk_ai.api.errors import ValidationAppError
from vendor_risk_ai.ingestion import extract as extract_mod
from vendor_risk_ai.ingestion.chunker import chunk_text
from vendor_risk_ai.ingestion.extract import extract_document, extract_text
from vendor_risk_ai.ingestion.normalize import normalize_text


# ----------------------------------------------------------------------------- extract
def test_extract_txt_and_md_utf8_with_replacement() -> None:
    """AC-003.2: .txt/.md decoded as UTF-8; invalid bytes replaced, not fatal."""
    assert extract_text("Résumé of risk".encode(), "notes.txt") == "Résumé of risk"
    assert extract_text(b"# Title\nbody", "README.MD") == "# Title\nbody"
    assert "�" in extract_text(b"bad \xff byte", "x.txt")


def test_extract_unsupported_type_raises_validation_error() -> None:
    """AC-003.2: unsupported extensions -> ValidationAppError (422)."""
    with pytest.raises(ValidationAppError) as ei:
        extract_text(b"PK\x03\x04", "report.docx")
    assert ei.value.status_code == 422
    assert ei.value.code == "unsupported_file_type"


def test_extract_empty_and_invalid_pdf() -> None:
    """AC-003.2: empty upload and corrupt PDFs are rejected with 422."""
    with pytest.raises(ValidationAppError):
        extract_text(b"", "a.txt")
    with pytest.raises(ValidationAppError) as ei:
        extract_text(b"%PDF-1.4 garbage", "broken.pdf")
    assert ei.value.code in {"invalid_pdf", "empty_document"}


def _pdf_with_text(text: str | None, pages: int = 1) -> bytes:
    """Tiny PDF built with pypdf; ``text=None`` gives blank pages (no text layer)."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=300, height=200)
        if text is None:
            continue
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_extract_pdf_text_layer() -> None:
    """AC-003.2: PDF text extracted page by page with pypdf."""
    data = _pdf_with_text("Going concern doubt disclosed", pages=2)
    result = extract_document(data, "filing.pdf")
    assert result.pages == 2
    assert result.text.count("Going concern doubt disclosed") == 2
    assert result.skipped_pages == [] and result.ocr_pages == []


def test_extract_pdf_without_text_layer_skips_when_no_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC-003.2: page with no text layer and no OCR available is skipped and noted."""
    monkeypatch.setattr(extract_mod, "ocr_available", lambda: False)
    with pytest.raises(ValidationAppError) as ei:
        extract_document(_pdf_with_text(None), "scan.pdf")
    assert ei.value.code == "empty_document"
    assert ei.value.details == {"skipped_pages": [1]}


def test_extract_pdf_uses_ocr_for_pages_without_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC-003.2: OCR (pytesseract) is used for pages lacking a text layer."""
    monkeypatch.setattr(extract_mod, "ocr_available", lambda: True)
    monkeypatch.setattr(extract_mod, "_ocr_page", lambda data, n: f"ocr text page {n}")
    result = extract_document(_pdf_with_text(None, pages=2), "scan.pdf")
    assert result.ocr_pages == [1, 2]
    assert "ocr text page 2" in result.text
    assert any("OCR" in n for n in result.notes)


# ----------------------------------------------------------------------------- normalize
def test_normalize_unicode_whitespace_controls_hyphenation() -> None:
    """AC-003.2: NFKC, control chars stripped, whitespace collapsed, hyphen breaks joined."""
    raw = "ﬁnancial  report\x00​ with  tabs\t\there\r\nliqui-\n  dity risk\n\n\n\nnext"
    out = normalize_text(raw)
    assert out == "financial report with tabs here\nliquidity risk\n\nnext"
    assert normalize_text("") == ""
    assert normalize_text("  \n\t ") == ""


# ----------------------------------------------------------------------------- chunker
def test_chunker_validates_arguments() -> None:
    """AC-003.2: overlap must be < size; size > 0."""
    with pytest.raises(ValueError):
        chunk_text("abc", 10, 10)
    with pytest.raises(ValueError):
        chunk_text("abc", 0, 0)
    with pytest.raises(ValueError):
        chunk_text("abc", 10, -1)
    assert chunk_text("   ", 10, 2) == []
    assert chunk_text("short text", 100, 10) == ["short text"]


def test_chunker_size_overlap_and_boundaries() -> None:
    """AC-003.2: chunks <= size, consecutive chunks overlap, cuts prefer sentence ends."""
    sentences = [f"Sentence number {i} talks about vendor risk item {i}." for i in range(40)]
    text = " ".join(sentences)
    chunks = chunk_text(text, size=200, overlap=50)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)
    # every chunk but the last ends on a sentence boundary
    assert all(c.endswith(".") for c in chunks[:-1])
    # overlap: the tail of chunk i reappears at the start of chunk i+1
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert b[:15] in a
    # no words are split at chunk starts
    words = set(text.split())
    assert all(c.split()[0] in words for c in chunks)
    # full coverage
    assert chunks[0].startswith("Sentence number 0") and chunks[-1].endswith("item 39.")


def test_chunker_hard_cuts_unbroken_text() -> None:
    """AC-003.2: text without boundaries is still chunked with the requested overlap."""
    chunks = chunk_text("x" * 250, size=100, overlap=20)
    assert [len(c) for c in chunks] == [100, 100, 90]
