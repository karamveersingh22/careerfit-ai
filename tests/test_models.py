"""Tests for the Part 1 document contracts."""

import pytest
from pydantic import ValidationError

from careerfit.models import (
    DocumentType,
    ExtractedDocument,
    ExtractedPage,
    JobProfile,
    ResumeProfile,
)


def test_document_computes_counts_and_combined_text() -> None:
    document = ExtractedDocument(
        document_type=DocumentType.JOB_DESCRIPTION,
        original_filename="role.pdf",
        pages=(
            ExtractedPage(page_number=1, text="Required skills", character_count=15),
        ),
    )

    assert document.page_count == 1
    assert document.character_count == 15
    assert document.text == "--- Page 1 ---\nRequired skills"


def test_page_number_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ExtractedPage(page_number=0, text="Invalid", character_count=7)


def test_unknown_profile_values_remain_null_or_empty() -> None:
    resume = ResumeProfile()
    job = JobProfile()

    assert resume.candidate_name is None
    assert resume.skills == []
    assert job.minimum_experience_years is None
    assert job.required_skills == []
