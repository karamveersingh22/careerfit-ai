"""Tests for safe PDF intake and page-aware extraction."""

import fitz
import pytest

from careerfit.document_engine import (
    MAX_PDF_BYTES,
    PDFValidationError,
    clean_extracted_text,
    extract_pdf,
    validate_pdf_upload,
)
from careerfit.models import DocumentType


def make_pdf(*page_texts: str) -> bytes:
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


def test_extracts_text_and_preserves_pages() -> None:
    content = make_pdf(
        "Resume page with Python and API development experience.",
        "Second page with education and project information.",
    )

    result = extract_pdf("candidate.PDF", content, DocumentType.RESUME)

    assert result.page_count == 2
    assert result.pages[0].page_number == 1
    assert "Python" in result.pages[0].text
    assert result.pages[1].page_number == 2
    assert "--- Page 1 ---" in result.text
    assert "--- Page 2 ---" in result.text


def test_cleaning_is_conservative() -> None:
    raw = "  First   line \r\n\r\n\r\n Second\tline \x00"
    assert clean_extracted_text(raw) == "First line\n\nSecond line"


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("resume.txt", b"not a pdf", ".pdf extension"),
        ("resume.pdf", b"", "empty"),
        ("resume.pdf", b"not a pdf", "not a valid PDF"),
    ],
)
def test_rejects_invalid_uploads(filename: str, content: bytes, message: str) -> None:
    with pytest.raises(PDFValidationError, match=message):
        validate_pdf_upload(filename, content)


def test_rejects_oversized_upload() -> None:
    content = b"%PDF-" + b"x" * MAX_PDF_BYTES
    with pytest.raises(PDFValidationError, match="10 MB"):
        validate_pdf_upload("large.pdf", content)


def test_rejects_image_only_or_empty_text_pdf() -> None:
    content = make_pdf("")
    with pytest.raises(PDFValidationError, match="image-only PDF"):
        extract_pdf("scan.pdf", content, DocumentType.RESUME)


def test_rejects_corrupt_pdf_after_signature_check() -> None:
    with pytest.raises(PDFValidationError, match="damaged"):
        extract_pdf("broken.pdf", b"%PDF-this-is-corrupt", DocumentType.RESUME)

