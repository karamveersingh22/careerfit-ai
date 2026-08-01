"""Page-safe, section-aware document chunking."""

from __future__ import annotations

import re

from .models import DocumentChunk, ExtractedDocument

SECTION_ALIASES = {
    "summary": "summary",
    "professional summary": "summary",
    "profile": "summary",
    "objective": "summary",
    "skills": "skills",
    "technical skills": "skills",
    "core skills": "skills",
    "experience": "experience",
    "work experience": "experience",
    "professional experience": "experience",
    "employment history": "experience",
    "education": "education",
    "academic background": "education",
    "projects": "projects",
    "personal projects": "projects",
    "certifications": "certifications",
    "certificates": "certifications",
    "responsibilities": "responsibilities",
    "requirements": "requirements",
    "required qualifications": "requirements",
    "preferred qualifications": "preferred qualifications",
    "about the role": "role",
    "job description": "role",
}


def _heading(line: str) -> str | None:
    normalized = re.sub(r"[^a-z ]", " ", line.casefold())
    normalized = " ".join(normalized.split())
    return SECTION_ALIASES.get(normalized)


def _split_long_text(text: str, max_characters: int) -> list[str]:
    if len(text) <= max_characters:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_characters:
            words = sentence.split()
            hard_part = ""
            for word in words:
                if hard_part and len(hard_part) + len(word) + 1 > max_characters:
                    if current:
                        chunks.append(current.strip())
                        current = ""
                    chunks.append(hard_part.strip())
                    hard_part = word
                else:
                    hard_part = f"{hard_part} {word}".strip()
            sentence = hard_part
        if current and len(current) + len(sentence) + 1 > max_characters:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def chunk_document(
    document: ExtractedDocument,
    analysis_id: str,
    max_characters: int = 700,
) -> list[DocumentChunk]:
    """Create chunks that never cross PDF pages and retain detected sections."""

    if max_characters < 100:
        raise ValueError("max_characters must be at least 100")

    chunks: list[DocumentChunk] = []
    chunk_number = 0
    current_section = "general"

    for page in document.pages:
        buffer: list[str] = []
        buffer_length = 0

        def flush() -> None:
            nonlocal chunk_number, buffer, buffer_length
            text = "\n".join(buffer).strip()
            if not text:
                return
            for part in _split_long_text(text, max_characters):
                chunk_number += 1
                chunk_id = (
                    f"{analysis_id}:{document.document_type.value}:"
                    f"p{page.page_number}:c{chunk_number}"
                )
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        analysis_id=analysis_id,
                        document_type=document.document_type,
                        section=current_section,
                        page=page.page_number,
                        text=part,
                    )
                )
            buffer = []
            buffer_length = 0

        for raw_line in page.text.splitlines():
            line = raw_line.strip()
            if not line:
                flush()
                continue
            detected_section = _heading(line)
            if detected_section:
                flush()
                current_section = detected_section
                continue
            projected = buffer_length + len(line) + (1 if buffer else 0)
            if buffer and projected > max_characters:
                flush()
            buffer.append(line)
            buffer_length += len(line) + (1 if buffer_length else 0)
        flush()

    return chunks
