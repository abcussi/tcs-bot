"""Text extraction from uploaded documents (AC-003.2).

PDF pages are read with pypdf; a page without a text layer is OCR'd with pdf2image + pytesseract
when both libraries *and* their binaries (poppler ``pdftoppm``, ``tesseract``) are available,
otherwise it is skipped and reported in ``ExtractionResult.skipped_pages``.
"""

from __future__ import annotations

import io
import shutil
from dataclasses import dataclass, field
from pathlib import PurePath

from tcs_bot.errors import ValidationAppError
from tcs_bot.logging_setup import get_logger

log = get_logger(__name__)

TEXT_EXTENSIONS = frozenset({".txt", ".md"})
PDF_EXTENSIONS = frozenset({".pdf"})
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS
_MIN_PAGE_CHARS = 3  # fewer non-space chars than this => treat as "no text layer"


@dataclass
class ExtractionResult:
    text: str
    pages: int = 0
    ocr_pages: list[int] = field(default_factory=list)  # 1-based page numbers
    skipped_pages: list[int] = field(default_factory=list)  # 1-based page numbers

    @property
    def notes(self) -> list[str]:
        notes = []
        if self.ocr_pages:
            notes.append(f"OCR used for page(s) {self.ocr_pages}")
        if self.skipped_pages:
            notes.append(f"No text layer and OCR unavailable for page(s) {self.skipped_pages}")
        return notes


def ocr_available() -> bool:
    """True when pdf2image + pytesseract import and poppler/tesseract binaries are on PATH."""
    try:
        import pdf2image  # noqa: F401
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return shutil.which("tesseract") is not None and shutil.which("pdftoppm") is not None


def _ocr_page(data: bytes, page_number: int) -> str:
    """OCR one 1-based page of a PDF. Raises on any failure (caller decides)."""
    import pytesseract
    from pdf2image import convert_from_bytes

    images = convert_from_bytes(data, dpi=300, first_page=page_number, last_page=page_number)
    return "\n".join(pytesseract.image_to_string(img) for img in images)


def _extract_pdf(data: bytes, filename: str) -> ExtractionResult:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = list(reader.pages)
    except (PdfReadError, ValueError, OSError) as exc:
        raise ValidationAppError(
            f"Could not read PDF '{filename}'", {"reason": type(exc).__name__}, code="invalid_pdf"
        ) from exc

    result = ExtractionResult(text="", pages=len(pages))
    can_ocr: bool | None = None  # resolved lazily, only if a page needs it
    texts: list[str] = []
    for number, page in enumerate(pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # pypdf can raise many things on malformed content streams
            log.warning("pdf_page_extract_failed", page=number, error_type=type(exc).__name__)
            page_text = ""
        if len(page_text.strip()) >= _MIN_PAGE_CHARS:
            texts.append(page_text)
            continue
        if can_ocr is None:
            can_ocr = ocr_available()
        if can_ocr:
            try:
                ocr_text = _ocr_page(data, number)
            except Exception as exc:
                log.warning("pdf_page_ocr_failed", page=number, error_type=type(exc).__name__)
                ocr_text = ""
            if ocr_text.strip():
                texts.append(ocr_text)
                result.ocr_pages.append(number)
                continue
        result.skipped_pages.append(number)

    if result.skipped_pages:
        log.warning("pdf_pages_skipped", filename=filename, pages=result.skipped_pages)
    result.text = "\n\n".join(texts)
    return result


def extract_document(data: bytes, filename: str) -> ExtractionResult:
    """Extract text plus page diagnostics. Raises ``ValidationAppError`` on bad input."""
    suffix = PurePath(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValidationAppError(
            f"Unsupported file type '{suffix or filename}'",
            {"supported": sorted(SUPPORTED_EXTENSIONS)},
            code="unsupported_file_type",
        )
    if not data:
        raise ValidationAppError("Uploaded file is empty", code="empty_document")

    if suffix in PDF_EXTENSIONS:
        result = _extract_pdf(data, filename)
    else:
        text = data.decode("utf-8", errors="replace")
        result = ExtractionResult(text=text.removeprefix("﻿"), pages=1)

    if not result.text.strip():
        raise ValidationAppError(
            f"No extractable text in '{filename}'",
            {"skipped_pages": result.skipped_pages},
            code="empty_document",
        )
    return result


def extract_text(data: bytes, filename: str) -> str:
    """Contract entry point (plan.md): bytes + filename -> raw text."""
    return extract_document(data, filename).text
