"""Hand-checkable tests for every deterministic scoring component."""

import pytest

from careerfit.models import (
    DocumentType,
    EducationEntry,
    EvidenceItem,
    ExtractedDocument,
    ExtractedPage,
    JobProfile,
    ProfileExtractionResult,
    ProjectEntry,
    ResumeProfile,
    SkillEvidence,
    WorkExperience,
)
from careerfit.scoring import (
    COMPONENT_WEIGHTS,
    READINESS_WEIGHTS,
    education_score,
    experience_score,
    responsibility_matches,
    responsibility_score,
    score_profiles,
)


def resume_document() -> ExtractedDocument:
    text = """SUMMARY
Software developer
SKILLS
Python JavaScript Django
EXPERIENCE
Developed REST APIs and reduced response time by 30 percent.
EDUCATION
Bachelor of Technology
PROJECTS
API platform
"""
    return ExtractedDocument(
        document_type=DocumentType.RESUME,
        original_filename="resume.pdf",
        pages=(ExtractedPage(page_number=1, text=text, character_count=len(text)),),
    )


def complete_resume() -> ResumeProfile:
    return ResumeProfile(
        candidate_name="Candidate",
        email="candidate@example.com",
        phone="1234567890",
        professional_summary=EvidenceItem(text="Software developer", source_page=1),
        skills=[
            SkillEvidence(name="Python", source_page=1),
            SkillEvidence(name="JS", source_page=1),
            SkillEvidence(name="Django", source_page=1),
            SkillEvidence(name="Git", source_page=1),
            SkillEvidence(name="SQL", source_page=1),
        ],
        work_experience=[
            WorkExperience(
                role="Developer",
                duration_years=1,
                responsibilities=[
                    EvidenceItem(
                        text="Developed REST APIs using Python and reduced response time by 30 percent.",
                        source_page=1,
                    )
                ],
                source_page=1,
            )
        ],
        total_experience_years=1,
        education=[
            EducationEntry(
                qualification="Bachelor of Technology",
                field_of_study="Computer Science",
                source_page=1,
            )
        ],
        projects=[
            ProjectEntry(
                name="API platform",
                description="Built an API used by 20 clients.",
                skills=["Python"],
                source_page=1,
            )
        ],
    )


def target_job() -> JobProfile:
    return JobProfile(
        job_title="Software intern",
        required_skills=[
            SkillEvidence(name="Python", source_page=1),
            SkillEvidence(name="JavaScript", source_page=1),
            SkillEvidence(name="Flask", source_page=1),
            SkillEvidence(name="Go", source_page=1),
        ],
        minimum_experience_years=2,
        responsibilities=[
            EvidenceItem(text="Develop REST APIs using Python", source_page=1)
        ],
        education_requirements=[
            EvidenceItem(
                text="Bachelor degree in Computer Science", source_page=1
            )
        ],
    )


@pytest.mark.parametrize(
    ("candidate", "required", "expected"),
    [
        (None, None, 100.0),
        (None, 2, 0.0),
        (1, 2, 50.0),
        (3, 2, 100.0),
    ],
)
def test_experience_formula(candidate, required, expected) -> None:
    assert experience_score(candidate, required) == expected


def test_responsibility_matching_returns_best_page_linked_evidence() -> None:
    matches = responsibility_matches(target_job().responsibilities, complete_resume())

    assert len(matches) == 1
    assert matches[0].resume_page == 1
    assert matches[0].score == 75.0
    assert responsibility_score(matches) == 75.0


def test_education_recognizes_equal_or_higher_degree_level() -> None:
    assert education_score(
        complete_resume().education,
        target_job().education_requirements,
    ) == 100.0


def test_absent_requirements_do_not_penalize_components() -> None:
    assert education_score([], []) == 100.0
    assert responsibility_score([]) == 100.0


def test_end_to_end_score_matches_published_weighted_formula() -> None:
    profiles = ProfileExtractionResult(
        model_name="test-model",
        resume_profile=complete_resume(),
        job_profile=target_job(),
    )

    result = score_profiles(profiles, resume_document())
    hand_calculated = round(
        sum(
            result.component_scores[name] * COMPONENT_WEIGHTS[name]
            for name in COMPONENT_WEIGHTS
        ),
        1,
    )

    assert result.component_scores["required_skills"] == 60.0
    assert result.component_scores["experience"] == 50.0
    assert result.component_scores["responsibilities"] == 75.0
    assert result.component_scores["education"] == 100.0
    assert result.overall_match_score == hand_calculated
    assert result.missing_skills == ["Go"]
    assert sum(READINESS_WEIGHTS.values()) == pytest.approx(1.0)


def test_readiness_is_separate_and_bounded() -> None:
    profiles = ProfileExtractionResult(
        model_name="test-model",
        resume_profile=complete_resume(),
        job_profile=target_job(),
    )
    result = score_profiles(profiles, resume_document())

    assert 0 <= result.ats_readiness_score <= 100
    assert len(result.readiness_checks) == 9
    assert result.component_scores["resume_quality"] == result.ats_readiness_score
