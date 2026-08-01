"""Validated data passed between CareerFit application layers."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field


class DocumentType(StrEnum):
    """The two document roles supported by the first version."""

    RESUME = "resume"
    JOB_DESCRIPTION = "job_description"


class ExtractedPage(BaseModel):
    """Text and provenance for one PDF page."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    text: str
    character_count: int = Field(ge=0)


class ExtractedDocument(BaseModel):
    """A page-aware, validated document produced by the document engine."""

    model_config = ConfigDict(frozen=True)

    document_type: DocumentType
    original_filename: str
    pages: tuple[ExtractedPage, ...]

    @computed_field
    @property
    def page_count(self) -> int:
        return len(self.pages)

    @computed_field
    @property
    def character_count(self) -> int:
        return sum(page.character_count for page in self.pages)

    @computed_field
    @property
    def text(self) -> str:
        """Return combined text while keeping visible page boundaries."""

        return "\n\n".join(
            f"--- Page {page.page_number} ---\n{page.text}" for page in self.pages
        )


class EvidenceItem(BaseModel):
    """A fact copied from a document with its source page."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    source_page: int = Field(ge=1)


class SkillEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    source_page: int = Field(ge=1)


class WorkExperience(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str | None = None
    company: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    duration_years: float | None = Field(default=None, ge=0)
    duration_is_estimated: bool = False
    responsibilities: list[EvidenceItem] = Field(default_factory=list)
    tools: list[SkillEvidence] = Field(default_factory=list)
    source_page: int = Field(ge=1)


class EducationEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    qualification: str | None = None
    field_of_study: str | None = None
    institution: str | None = None
    graduation_date: str | None = None
    source_page: int = Field(ge=1)


class ProjectEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str | None = None
    description: str | None = None
    skills: list[str] = Field(default_factory=list)
    source_page: int = Field(ge=1)


class ResumeProfile(BaseModel):
    """Structured evidence extracted from a résumé, never inferred beyond text."""

    model_config = ConfigDict(frozen=True)

    candidate_name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    professional_summary: EvidenceItem | None = None
    skills: list[SkillEvidence] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    total_experience_years: float | None = Field(default=None, ge=0)
    total_experience_is_estimated: bool = False
    education: list[EducationEntry] = Field(default_factory=list)
    certifications: list[EvidenceItem] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)


class JobProfile(BaseModel):
    """Structured requirements extracted from a job description."""

    model_config = ConfigDict(frozen=True)

    job_title: str | None = None
    company: str | None = None
    employment_type: str | None = None
    location: str | None = None
    required_skills: list[SkillEvidence] = Field(default_factory=list)
    preferred_skills: list[SkillEvidence] = Field(default_factory=list)
    minimum_experience_years: float | None = Field(default=None, ge=0)
    responsibilities: list[EvidenceItem] = Field(default_factory=list)
    education_requirements: list[EvidenceItem] = Field(default_factory=list)
    tools: list[SkillEvidence] = Field(default_factory=list)
    domain_knowledge: list[EvidenceItem] = Field(default_factory=list)


class ProfileExtractionResult(BaseModel):
    """Part 2 output consumed by later comparison and presentation layers."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    resume_profile: ResumeProfile
    job_profile: JobProfile


class MatchType(StrEnum):
    EXACT = "exact"
    EQUIVALENT = "equivalent"
    RELATED = "related"
    MISSING = "missing"


class SkillMatch(BaseModel):
    """Transparent credit for one job skill requirement."""

    model_config = ConfigDict(frozen=True)

    requirement: str
    job_page: int
    match_type: MatchType
    credit: float = Field(ge=0, le=1)
    resume_skill: str | None = None
    resume_page: int | None = Field(default=None, ge=1)


class ResponsibilityMatch(BaseModel):
    """Best lexical evidence for one job responsibility."""

    model_config = ConfigDict(frozen=True)

    requirement: str
    job_page: int
    resume_evidence: str | None = None
    resume_page: int | None = Field(default=None, ge=1)
    score: float = Field(ge=0, le=100)
    method: str = "lexical"
    section: str | None = None
    chunk_id: str | None = None


class ReadinessCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    score: float = Field(ge=0, le=100)
    weight: float = Field(gt=0, le=1)
    explanation: str


class ScoringResult(BaseModel):
    """Deterministic Part 3 result; no model-generated scores."""

    model_config = ConfigDict(frozen=True)

    overall_match_score: float = Field(ge=0, le=100)
    ats_readiness_score: float = Field(ge=0, le=100)
    component_scores: dict[str, float]
    component_weights: dict[str, float]
    skill_matches: list[SkillMatch]
    responsibility_matches: list[ResponsibilityMatch]
    readiness_checks: list[ReadinessCheck]
    matched_skills: list[str]
    missing_skills: list[str]
    limitations: list[str]


class DocumentChunk(BaseModel):
    """One page-bound text unit stored in the vector database."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    analysis_id: str
    document_type: DocumentType
    section: str
    page: int = Field(ge=1)
    text: str = Field(min_length=1)


class RetrievalHit(BaseModel):
    """A filtered Chroma result with normalized cosine similarity."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    text: str
    page: int = Field(ge=1)
    section: str
    similarity: float = Field(ge=0, le=1)


class RagAnalysisResult(BaseModel):
    """Part 4 index and retrieval summary."""

    model_config = ConfigDict(frozen=True)

    analysis_id: str
    embedding_model: str
    embedding_dimension: int = Field(gt=0)
    chunk_count: int = Field(ge=0)
    semantic_responsibility_matches: list[ResponsibilityMatch]


class GuidanceItem(BaseModel):
    """One grounded strength or gap with evidence from both document roles."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    resume_evidence: str | None = None
    resume_page: int | None = Field(default=None, ge=1)
    job_evidence: str = Field(min_length=1)
    job_page: int = Field(ge=1)


class ResumeRewrite(BaseModel):
    """A truthful rewrite of an existing résumé statement."""

    model_config = ConfigDict(frozen=True)

    original_text: str = Field(min_length=1)
    improved_text: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    resume_page: int = Field(ge=1)
    target_job_evidence: str = Field(min_length=1)
    job_page: int = Field(ge=1)


class LearningRecommendation(BaseModel):
    """A learning action tied to an explicit target-job requirement."""

    model_config = ConfigDict(frozen=True)

    topic: str = Field(min_length=1)
    why_it_matters: str = Field(min_length=1)
    suggested_action: str = Field(min_length=1)
    job_evidence: str = Field(min_length=1)
    job_page: int = Field(ge=1)


class InterviewQuestion(BaseModel):
    """A grounded interview question and the evidence used to create it."""

    model_config = ConfigDict(frozen=True)

    category: str = Field(pattern="^(technical|resume|gap)$")
    question: str = Field(min_length=1)
    preparation_tip: str = Field(min_length=1)
    job_evidence: str = Field(min_length=1)
    job_page: int = Field(ge=1)
    resume_evidence: str | None = None
    resume_page: int | None = Field(default=None, ge=1)


class CareerGuidanceResult(BaseModel):
    """Validated Part 5 guidance; it never supplies or modifies scores."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    strengths: list[GuidanceItem] = Field(default_factory=list)
    gaps: list[GuidanceItem] = Field(default_factory=list)
    rewrites: list[ResumeRewrite] = Field(default_factory=list)
    learning_plan: list[LearningRecommendation] = Field(default_factory=list)
    interview_questions: list[InterviewQuestion] = Field(default_factory=list)
