"""Published, deterministic CareerFit match and résumé-readiness scoring."""

from __future__ import annotations

import re

from .models import (
    EducationEntry,
    EvidenceItem,
    ExtractedDocument,
    MatchType,
    ProfileExtractionResult,
    ReadinessCheck,
    ResponsibilityMatch,
    ResumeProfile,
    ScoringResult,
    SkillEvidence,
)
from .skill_matching import (
    collect_resume_skills,
    match_skill_list,
    normalize_skill,
    skill_component_score,
)

COMPONENT_WEIGHTS: dict[str, float] = {
    "required_skills": 0.40,
    "experience": 0.20,
    "responsibilities": 0.15,
    "preferred_skills": 0.10,
    "education": 0.10,
    "resume_quality": 0.05,
}

READINESS_WEIGHTS: dict[str, float] = {
    "Contact details": 0.15,
    "Professional summary": 0.10,
    "Skills section": 0.15,
    "Experience or projects": 0.20,
    "Education": 0.10,
    "Clear headings": 0.10,
    "Action verbs": 0.05,
    "Quantified achievements": 0.10,
    "Relevant keywords": 0.05,
}

STOP_WORDS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "in", "is",
    "of", "on", "or", "that", "the", "their", "to", "using", "with", "will",
    "work", "working", "ability", "candidate", "responsible", "including",
}

ACTION_VERBS = {
    "achieved", "automated", "built", "created", "delivered", "designed",
    "developed", "implemented", "improved", "increased", "launched", "led",
    "managed", "optimized", "reduced", "resolved", "streamlined", "tested",
}

DEGREE_LEVELS = {
    "high school": 1,
    "secondary": 1,
    "associate": 2,
    "diploma": 2,
    "bachelor": 3,
    "bachelors": 3,
    "undergraduate": 3,
    "btech": 3,
    "b tech": 3,
    "master": 4,
    "masters": 4,
    "postgraduate": 4,
    "mtech": 4,
    "m tech": 4,
    "phd": 5,
    "doctorate": 5,
}


def experience_score(candidate_years: float | None, required_years: float | None) -> float:
    if required_years is None or required_years <= 0:
        return 100.0
    if candidate_years is None:
        return 0.0
    return round(min(candidate_years / required_years, 1.0) * 100, 1)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in normalize_skill(text).split()
        if len(token) > 2 and token not in STOP_WORDS
    }


def responsibility_matches(
    requirements: list[EvidenceItem],
    resume: ResumeProfile,
) -> list[ResponsibilityMatch]:
    evidence = [
        item
        for experience in resume.work_experience
        for item in experience.responsibilities
    ]
    evidence.extend(
        EvidenceItem(text=project.description, source_page=project.source_page)
        for project in resume.projects
        if project.description
    )

    matches: list[ResponsibilityMatch] = []
    for requirement in requirements:
        required_tokens = _tokens(requirement.text)
        best_score = 0.0
        best_evidence: EvidenceItem | None = None
        for item in evidence:
            if not required_tokens:
                score = 0.0
            else:
                score = len(required_tokens & _tokens(item.text)) / len(required_tokens) * 100
            if score > best_score:
                best_score = score
                best_evidence = item
        matches.append(
            ResponsibilityMatch(
                requirement=requirement.text,
                job_page=requirement.source_page,
                resume_evidence=best_evidence.text if best_evidence else None,
                resume_page=best_evidence.source_page if best_evidence else None,
                score=round(best_score, 1),
            )
        )
    return matches


def responsibility_score(matches: list[ResponsibilityMatch]) -> float:
    if not matches:
        return 100.0
    return round(sum(match.score for match in matches) / len(matches), 1)


def _degree_level(text: str) -> int | None:
    normalized = normalize_skill(text)
    levels = [level for label, level in DEGREE_LEVELS.items() if label in normalized]
    return max(levels) if levels else None


def education_score(
    education: list[EducationEntry], requirements: list[EvidenceItem]
) -> float:
    if not requirements:
        return 100.0
    if not education:
        return 0.0

    candidate_texts = [
        " ".join(
            value
            for value in (entry.qualification, entry.field_of_study, entry.institution)
            if value
        )
        for entry in education
    ]
    candidate_level = max(
        (level for text in candidate_texts if (level := _degree_level(text)) is not None),
        default=None,
    )

    requirement_scores: list[float] = []
    for requirement in requirements:
        required_level = _degree_level(requirement.text)
        level_credit = 100.0 if (
            required_level is not None
            and candidate_level is not None
            and candidate_level >= required_level
        ) else 0.0
        required_tokens = _tokens(requirement.text)
        lexical_credit = max(
            (
                len(required_tokens & _tokens(text)) / len(required_tokens) * 100
                if required_tokens else 0.0
            )
            for text in candidate_texts
        )
        requirement_scores.append(max(level_credit, lexical_credit))
    return round(sum(requirement_scores) / len(requirement_scores), 1)


def _resume_skill_evidence(resume: ResumeProfile) -> list[SkillEvidence]:
    evidence = list(resume.skills)
    evidence.extend(tool for exp in resume.work_experience for tool in exp.tools)
    evidence.extend(
        SkillEvidence(name=skill, source_page=project.source_page)
        for project in resume.projects
        for skill in project.skills
    )
    return collect_resume_skills(evidence)


def readiness_score(
    resume: ResumeProfile,
    resume_document: ExtractedDocument,
    required_skill_score: float,
) -> tuple[float, list[ReadinessCheck]]:
    raw_text = resume_document.text
    achievement_text = " ".join(
        [
            item.text
            for experience in resume.work_experience
            for item in experience.responsibilities
        ]
        + [project.description or "" for project in resume.projects]
    ).casefold()

    contact_count = int(bool(resume.email)) + int(bool(resume.phone))
    heading_names = ("summary", "skills", "experience", "education", "projects")
    headings_found = sum(
        bool(re.search(rf"(?im)^\s*{heading}\s*:?\s*$", raw_text))
        for heading in heading_names
    )
    action_count = sum(
        bool(re.search(rf"\b{re.escape(verb)}\b", achievement_text))
        for verb in ACTION_VERBS
    )
    quantified_count = len(
        re.findall(
            r"\b\d+(?:\.\d+)?\s*(?:%|percent|x|\+|users|clients|hours|days)\b",
            achievement_text,
        )
    )

    raw_checks = {
        "Contact details": (
            contact_count / 2 * 100,
            "Email and phone are checked separately.",
        ),
        "Professional summary": (
            100.0 if resume.professional_summary else 0.0,
            "A clearly extracted professional summary is present." if resume.professional_summary else "No professional summary was identified.",
        ),
        "Skills section": (
            min(len(resume.skills) / 5, 1) * 100,
            f"{len(resume.skills)} explicit skills were extracted.",
        ),
        "Experience or projects": (
            100.0 if resume.work_experience else (60.0 if resume.projects else 0.0),
            "Work experience is present." if resume.work_experience else "Projects provide partial evidence when work experience is absent.",
        ),
        "Education": (
            100.0 if resume.education else 0.0,
            "Education information is present." if resume.education else "No education information was identified.",
        ),
        "Clear headings": (
            min(headings_found / 4, 1) * 100,
            f"{headings_found} standard section headings were detected.",
        ),
        "Action verbs": (
            min(action_count / 3, 1) * 100,
            f"{action_count} distinct action verbs were detected.",
        ),
        "Quantified achievements": (
            min(quantified_count / 2, 1) * 100,
            f"{quantified_count} numeric result indicators were detected.",
        ),
        "Relevant keywords": (
            required_skill_score,
            "Uses the deterministic required-skill coverage score.",
        ),
    }

    checks = [
        ReadinessCheck(
            name=name,
            score=round(score, 1),
            weight=READINESS_WEIGHTS[name],
            explanation=explanation,
        )
        for name, (score, explanation) in raw_checks.items()
    ]
    total = round(sum(check.score * check.weight for check in checks), 1)
    return total, checks


def score_profiles(
    profiles: ProfileExtractionResult,
    resume_document: ExtractedDocument,
    responsibility_evidence_override: list[ResponsibilityMatch] | None = None,
) -> ScoringResult:
    resume = profiles.resume_profile
    job = profiles.job_profile
    resume_skills = _resume_skill_evidence(resume)

    required_matches = match_skill_list(job.required_skills, resume_skills)
    preferred_matches = match_skill_list(job.preferred_skills, resume_skills)
    all_skill_matches = required_matches + preferred_matches
    required_score = skill_component_score(required_matches)
    preferred_score = skill_component_score(preferred_matches)

    responsibility_evidence = (
        responsibility_evidence_override
        if responsibility_evidence_override is not None
        else responsibility_matches(job.responsibilities, resume)
    )
    readiness, readiness_checks = readiness_score(
        resume, resume_document, required_score
    )

    component_scores = {
        "required_skills": required_score,
        "experience": experience_score(
            resume.total_experience_years, job.minimum_experience_years
        ),
        "responsibilities": responsibility_score(responsibility_evidence),
        "preferred_skills": preferred_score,
        "education": education_score(
            resume.education, job.education_requirements
        ),
        "resume_quality": readiness,
    }
    overall = round(
        sum(component_scores[name] * weight for name, weight in COMPONENT_WEIGHTS.items()),
        1,
    )

    matched = [
        match.requirement
        for match in all_skill_matches
        if match.match_type != MatchType.MISSING
    ]
    missing = [
        match.requirement
        for match in all_skill_matches
        if match.match_type == MatchType.MISSING
    ]

    return ScoringResult(
        overall_match_score=overall,
        ats_readiness_score=readiness,
        component_scores=component_scores,
        component_weights=COMPONENT_WEIGHTS,
        skill_matches=all_skill_matches,
        responsibility_matches=responsibility_evidence,
        readiness_checks=readiness_checks,
        matched_skills=matched,
        missing_skills=missing,
        limitations=[
            "This is a published CareerFit formula, not a proprietary ATS score.",
            "Responsibility evidence uses the stronger of transparent word overlap and filtered semantic retrieval.",
            "A missing résumé mention means missing evidence, not necessarily missing real-world ability.",
            "Experience currently compares stated total experience with the stated minimum.",
        ],
    )
