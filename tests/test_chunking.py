"""Tests for page-safe, section-aware chunk construction."""

from careerfit.chunking import chunk_document
from careerfit.models import DocumentType, ExtractedDocument, ExtractedPage


def test_chunks_preserve_analysis_source_section_and_page() -> None:
    document = ExtractedDocument(
        document_type=DocumentType.RESUME,
        original_filename="resume.pdf",
        pages=(
            ExtractedPage(
                page_number=1,
                text="SKILLS\nPython and SQL\n\nEXPERIENCE\nDeveloped APIs",
                character_count=52,
            ),
            ExtractedPage(
                page_number=2,
                text="EDUCATION\nBachelor of Technology",
                character_count=32,
            ),
        ),
    )

    chunks = chunk_document(document, "analysis-123")

    assert [chunk.section for chunk in chunks] == [
        "skills",
        "experience",
        "education",
    ]
    assert [chunk.page for chunk in chunks] == [1, 1, 2]
    assert all(chunk.analysis_id == "analysis-123" for chunk in chunks)
    assert all(chunk.document_type == DocumentType.RESUME for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_chunks_never_cross_pages_or_size_limit() -> None:
    document = ExtractedDocument(
        document_type=DocumentType.JOB_DESCRIPTION,
        original_filename="job.pdf",
        pages=(
            ExtractedPage(
                page_number=1,
                text="REQUIREMENTS\n" + "Python development " * 30,
                character_count=570,
            ),
            ExtractedPage(
                page_number=2,
                text="RESPONSIBILITIES\nBuild reliable APIs",
                character_count=37,
            ),
        ),
    )

    chunks = chunk_document(document, "analysis-1", max_characters=120)

    assert all(len(chunk.text) <= 120 for chunk in chunks)
    assert {chunk.page for chunk in chunks} == {1, 2}
    assert all("Build reliable APIs" not in chunk.text for chunk in chunks if chunk.page == 1)


def test_rejects_tiny_chunk_limit() -> None:
    document = ExtractedDocument(
        document_type=DocumentType.RESUME,
        original_filename="resume.pdf",
        pages=(),
    )

    try:
        chunk_document(document, "analysis", max_characters=50)
    except ValueError as exc:
        assert "at least 100" in str(exc)
    else:
        raise AssertionError("Expected a ValueError")
