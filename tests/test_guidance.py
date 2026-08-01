"""Tests for grounded, schema-constrained Part 5 career guidance."""

from types import SimpleNamespace

import pytest

from careerfit.guidance import (
    GeminiGuidanceGenerator,
    GuidanceError,
    build_guidance_prompt,
)
from careerfit.models import (
    CareerGuidanceResult,
    DocumentType,
    ExtractedDocument,
    ExtractedPage,
    JobProfile,
    ProfileExtractionResult,
    ResumeProfile,
)
from careerfit.scoring import score_profiles


class FakeModels:
    def __init__(self, outputs=None, error=None):
        self.outputs = list(outputs or [])
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.outputs.pop(0))


def inputs():
    resume = ExtractedDocument(
        document_type=DocumentType.RESUME,
        original_filename="resume.pdf",
        pages=(
            ExtractedPage(
                page_number=1,
                text="Built REST APIs using Python for customer workflows.",
                character_count=52,
            ),
        ),
    )
    job = ExtractedDocument(
        document_type=DocumentType.JOB_DESCRIPTION,
        original_filename="job.pdf",
        pages=(
            ExtractedPage(
                page_number=1,
                text="Develop Python APIs and automated workflows.",
                character_count=44,
            ),
        ),
    )
    profiles = ProfileExtractionResult(
        model_name="test-model",
        resume_profile=ResumeProfile(
            skills=[{"name": "Python", "source_page": 1}]
        ),
        job_profile=JobProfile(
            required_skills=[{"name": "Python", "source_page": 1}]
        ),
    )
    return profiles, score_profiles(profiles, resume), resume, job


def valid_guidance() -> str:
    return CareerGuidanceResult(
        model_name="test-model",
        strengths=[
            {
                "title": "Relevant Python experience",
                "explanation": "The résumé contains directly relevant evidence.",
                "resume_evidence": "Built REST APIs using Python",
                "resume_page": 1,
                "job_evidence": "Develop Python APIs",
                "job_page": 1,
            }
        ],
        gaps=[],
        rewrites=[
            {
                "original_text": "Built REST APIs using Python",
                "improved_text": "Built Python REST APIs for customer workflows.",
                "reason": "Makes the relevant technology prominent.",
                "resume_page": 1,
                "target_job_evidence": "Develop Python APIs",
                "job_page": 1,
            }
        ],
        learning_plan=[],
        interview_questions=[],
    ).model_dump_json()


def test_generates_guidance_and_verifies_page_quotes() -> None:
    profiles, score, resume, job = inputs()
    models = FakeModels([valid_guidance()])
    generator = GeminiGuidanceGenerator(
        api_key="test", model="test-model", client=SimpleNamespace(models=models)
    )

    result = generator.generate(profiles, score, resume, job)

    assert result.strengths[0].resume_page == 1
    assert result.rewrites[0].improved_text.startswith("Built Python")
    assert models.calls[0]["config"].response_schema is CareerGuidanceResult


def test_retries_guidance_with_an_invented_quote() -> None:
    profiles, score, resume, job = inputs()
    invented = CareerGuidanceResult.model_validate_json(valid_guidance()).model_copy(
        update={
            "strengths": [
                CareerGuidanceResult.model_validate_json(valid_guidance())
                .strengths[0]
                .model_copy(update={"resume_evidence": "Invented achievement"})
            ]
        }
    )
    models = FakeModels([invented.model_dump_json(), valid_guidance()])
    generator = GeminiGuidanceGenerator(
        api_key="test", model="test-model", client=SimpleNamespace(models=models)
    )

    result = generator.generate(profiles, score, resume, job)

    assert result.strengths[0].resume_evidence == "Built REST APIs using Python"
    assert len(models.calls) == 2


def test_provider_failure_does_not_retry() -> None:
    profiles, score, resume, job = inputs()
    models = FakeModels(error=RuntimeError("provider unavailable"))
    generator = GeminiGuidanceGenerator(
        api_key="test", model="test-model", client=SimpleNamespace(models=models)
    )

    with pytest.raises(GuidanceError, match="could not generate"):
        generator.generate(profiles, score, resume, job)

    assert len(models.calls) == 1


def test_prompt_prohibits_score_changes_and_invented_resume_facts() -> None:
    profiles, score, resume, job = inputs()
    prompt = build_guidance_prompt(profiles, score, resume, job)

    assert "Never recalculate, change" in prompt
    assert "never invent numbers" in prompt
    assert "verbatim excerpts" in prompt
