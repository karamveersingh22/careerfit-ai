"""Part 6 deterministic evaluation over synthetic résumé/JD pairs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .models import (
    DocumentType,
    ExtractedDocument,
    ExtractedPage,
    JobProfile,
    ProfileExtractionResult,
    ResumeProfile,
)
from .scoring import score_profiles


class EvaluationBand(StrEnum):
    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"


@dataclass(frozen=True)
class DomainSpec:
    slug: str
    title: str
    required_skills: tuple[str, ...]
    preferred_skills: tuple[str, ...]
    responsibilities: tuple[str, str]


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    domain: str
    expected_band: EvaluationBand
    profiles: ProfileExtractionResult
    resume_document: ExtractedDocument


@dataclass(frozen=True)
class EvaluationCaseResult:
    case_id: str
    domain: str
    expected_band: EvaluationBand
    actual_band: EvaluationBand
    overall_score: float
    readiness_score: float
    matched_skills: int
    missing_skills: int

    @property
    def passed(self) -> bool:
        return self.expected_band == self.actual_band


@dataclass(frozen=True)
class EvaluationReport:
    results: tuple[EvaluationCaseResult, ...]

    @property
    def case_count(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def band_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return round(self.passed_count / self.case_count * 100, 1)

    def average_score(self, band: EvaluationBand) -> float:
        values = [
            result.overall_score
            for result in self.results
            if result.expected_band == band
        ]
        return round(sum(values) / len(values), 1) if values else 0.0


DOMAIN_SPECS: tuple[DomainSpec, ...] = (
    DomainSpec(
        "backend", "Backend developer",
        ("Python", "FastAPI", "PostgreSQL", "REST APIs", "Git"),
        ("Docker", "AWS"),
        ("Build reliable Python APIs", "Design secure database workflows"),
    ),
    DomainSpec(
        "frontend", "Frontend developer",
        ("JavaScript", "TypeScript", "React", "HTML", "CSS"),
        ("Next.js", "Jest"),
        ("Build accessible React interfaces", "Test responsive user workflows"),
    ),
    DomainSpec(
        "data_science", "Data scientist",
        ("Python", "SQL", "Machine Learning", "Pandas", "Statistics"),
        ("TensorFlow", "NLP"),
        ("Train machine learning models", "Analyze structured business data"),
    ),
    DomainSpec(
        "cloud", "Cloud engineer",
        ("AWS", "Docker", "Kubernetes", "Terraform", "Linux"),
        ("CI/CD", "Azure"),
        ("Deploy containerized cloud services", "Automate infrastructure provisioning"),
    ),
    DomainSpec(
        "mobile", "Mobile developer",
        ("Kotlin", "Android", "REST APIs", "Git", "SQLite"),
        ("Firebase", "Java"),
        ("Build reliable Android applications", "Integrate mobile API workflows"),
    ),
    DomainSpec(
        "security", "Security analyst",
        ("Python", "Linux", "SQL", "Networking", "Git"),
        ("SIEM", "AWS"),
        ("Investigate security incidents", "Automate vulnerability reporting"),
    ),
    DomainSpec(
        "data_engineering", "Data engineer",
        ("Python", "SQL", "PostgreSQL", "Apache Spark", "ETL"),
        ("Airflow", "AWS"),
        ("Build reliable data pipelines", "Validate large analytical datasets"),
    ),
    DomainSpec(
        "fullstack", "Full-stack developer",
        ("JavaScript", "TypeScript", "React", "Node.js", "MongoDB"),
        ("Next.js", "Docker"),
        ("Build full-stack web applications", "Design authenticated API workflows"),
    ),
)


def score_band(score: float) -> EvaluationBand:
    """Convert an overall score into the published evaluation bands."""

    if score >= 75:
        return EvaluationBand.STRONG
    if score >= 45:
        return EvaluationBand.MEDIUM
    return EvaluationBand.WEAK


def _document(kind: DocumentType, filename: str, text: str) -> ExtractedDocument:
    return ExtractedDocument(
        document_type=kind,
        original_filename=filename,
        pages=(
            ExtractedPage(
                page_number=1,
                text=text,
                character_count=len(text),
            ),
        ),
    )


def _build_case(spec: DomainSpec, band: EvaluationBand) -> EvaluationCase:
    required = [
        {"name": skill, "source_page": 1} for skill in spec.required_skills
    ]
    preferred = [
        {"name": skill, "source_page": 1} for skill in spec.preferred_skills
    ]
    job = JobProfile(
        job_title=spec.title,
        required_skills=required,
        preferred_skills=preferred,
        minimum_experience_years=2,
        responsibilities=[
            {"text": text, "source_page": 1} for text in spec.responsibilities
        ],
        education_requirements=[
            {"text": "Bachelor degree in Computer Science", "source_page": 1}
        ],
    )

    if band == EvaluationBand.STRONG:
        resume_skill_names = [*spec.required_skills, *spec.preferred_skills]
        years = 3.0
        responsibility_texts = list(spec.responsibilities)
        has_education = True
        summary = {"text": f"Experienced {spec.title}.", "source_page": 1}
    elif band == EvaluationBand.MEDIUM:
        resume_skill_names = [*spec.required_skills[:3], spec.preferred_skills[0]]
        years = 1.0
        responsibility_texts = [spec.responsibilities[0]]
        has_education = True
        summary = None
    else:
        resume_skill_names = ["Microsoft Word"]
        years = None
        responsibility_texts = []
        has_education = False
        summary = None

    work_experience = []
    if responsibility_texts:
        work_experience = [
            {
                "role": spec.title,
                "company": "Synthetic Labs",
                "duration_years": years,
                "responsibilities": [
                    {"text": text, "source_page": 1}
                    for text in responsibility_texts
                ],
                "tools": [
                    {"name": skill, "source_page": 1}
                    for skill in resume_skill_names
                ],
                "source_page": 1,
            }
        ]

    education = (
        [
            {
                "qualification": "Bachelor degree",
                "field_of_study": "Computer Science",
                "institution": "Synthetic University",
                "source_page": 1,
            }
        ]
        if has_education else []
    )
    resume = ResumeProfile(
        candidate_name=f"Synthetic {band.value.title()} Candidate",
        email="candidate@example.com",
        phone="5550100" if band != EvaluationBand.WEAK else None,
        professional_summary=summary,
        skills=[
            {"name": skill, "source_page": 1} for skill in resume_skill_names
        ],
        work_experience=work_experience,
        total_experience_years=years,
        education=education,
    )

    resume_text = "\n".join(
        [
            resume.candidate_name or "Synthetic candidate",
            resume.email or "",
            resume.phone or "",
            "Summary" if summary else "Objective",
            summary["text"] if summary else "Seeking a new opportunity.",
            "Skills" if band != EvaluationBand.WEAK else "Interests",
            ", ".join(resume_skill_names),
            "Experience" if work_experience else "Activities",
            *responsibility_texts,
            "Education" if has_education else "",
            "Bachelor degree in Computer Science" if has_education else "",
            "Improved a workflow for 25 users." if band == EvaluationBand.STRONG else "",
        ]
    )
    profiles = ProfileExtractionResult(
        model_name="synthetic-fixture-v1",
        resume_profile=resume,
        job_profile=job,
    )
    return EvaluationCase(
        case_id=f"{spec.slug}-{band.value}",
        domain=spec.title,
        expected_band=band,
        profiles=profiles,
        resume_document=_document(
            DocumentType.RESUME,
            f"{spec.slug}-{band.value}-resume.pdf",
            resume_text,
        ),
    )


def build_synthetic_cases() -> tuple[EvaluationCase, ...]:
    """Build 24 balanced cases: eight domains across all three fit bands."""

    return tuple(
        _build_case(spec, band)
        for spec in DOMAIN_SPECS
        for band in EvaluationBand
    )


def run_synthetic_evaluation() -> EvaluationReport:
    """Run the production scoring engine over every synthetic case."""

    results: list[EvaluationCaseResult] = []
    for case in build_synthetic_cases():
        score = score_profiles(case.profiles, case.resume_document)
        results.append(
            EvaluationCaseResult(
                case_id=case.case_id,
                domain=case.domain,
                expected_band=case.expected_band,
                actual_band=score_band(score.overall_match_score),
                overall_score=score.overall_match_score,
                readiness_score=score.ats_readiness_score,
                matched_skills=len(score.matched_skills),
                missing_skills=len(score.missing_skills),
            )
        )
    return EvaluationReport(results=tuple(results))


if __name__ == "__main__":
    report = run_synthetic_evaluation()
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status:4} {result.case_id:28} score={result.overall_score:5.1f} "
            f"expected={result.expected_band.value:6} actual={result.actual_band.value}"
        )
    print(
        f"\n{report.passed_count}/{report.case_count} passed "
        f"({report.band_accuracy:.1f}% band accuracy)"
    )
