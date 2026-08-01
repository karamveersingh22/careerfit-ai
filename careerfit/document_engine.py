"""Safe, page-aware PDF validation and text extraction."""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from .models import DocumentType, ExtractedDocument, ExtractedPage

MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 50
MIN_DOCUMENT_CHARACTERS = 40
PDF_SIGNATURE = b"%PDF-"


class PDFValidationError(ValueError):
    """A safe, user-facing validation failure."""


def validate_pdf_upload(filename: str, content: bytes) -> None:
    """Perform inexpensive checks before opening untrusted PDF bytes."""

    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise PDFValidationError("Please upload a file with a .pdf extension.")
    if not content:
        raise PDFValidationError("The uploaded PDF is empty.")
    if len(content) > MAX_PDF_BYTES:
        raise PDFValidationError("The PDF is larger than the 10 MB limit.")
    if not content.startswith(PDF_SIGNATURE):
        raise PDFValidationError("The uploaded file is not a valid PDF.")


def clean_extracted_text(text: str) -> str:
    """Normalize harmless whitespace without changing document meaning."""

    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    cleaned_lines: list[str] = []
    blank_pending = False
    for line in lines:
        if line:
            if blank_pending and cleaned_lines:
                cleaned_lines.append("")
            cleaned_lines.append(line)
            blank_pending = False
        elif cleaned_lines:
            blank_pending = True

    return "\n".join(cleaned_lines).strip()


def extract_pdf(
    filename: str,
    content: bytes,
    document_type: DocumentType,
) -> ExtractedDocument:
    """Validate a PDF and extract text while retaining page provenance."""

    validate_pdf_upload(filename, content)

    try:
        pdf = fitz.open(stream=content, filetype="pdf")
    except (fitz.FileDataError, RuntimeError) as exc:
        raise PDFValidationError("The PDF is damaged or cannot be opened.") from exc

    try:
        if pdf.needs_pass:
            raise PDFValidationError(
                "Password-protected PDFs are not supported in this version."
            )
        if pdf.page_count == 0:
            raise PDFValidationError("The PDF contains no pages.")
        if pdf.page_count > MAX_PDF_PAGES:
            raise PDFValidationError(
                f"The PDF has more than the {MAX_PDF_PAGES}-page limit."
            )

        pages: list[ExtractedPage] = []
        for index, page in enumerate(pdf, start=1):
            text = clean_extracted_text(page.get_text("text", sort=True))
            pages.append(
                ExtractedPage(
                    page_number=index,
                    text=text,
                    character_count=len(text),
                )
            )

        total_characters = sum(page.character_count for page in pages)
        if total_characters < MIN_DOCUMENT_CHARACTERS:
            raise PDFValidationError(
                "No readable text was found. This may be an image-only PDF; OCR "
                "will be added in a later version."
            )

        return ExtractedDocument(
            document_type=document_type,
            original_filename=Path(filename).name,
            pages=tuple(pages),
        )
    finally:
        pdf.close()

