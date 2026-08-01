"""Deterministic skill normalization and transparent equivalence rules."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import MatchType, SkillEvidence, SkillMatch


ALIAS_GROUPS: dict[str, set[str]] = {
    "amazon web services": {"aws"},
    "artificial intelligence": {"ai"},
    "continuous integration continuous delivery": {"ci cd", "cicd"},
    "google cloud platform": {"gcp", "google cloud"},
    "javascript": {"js", "ecmascript"},
    "machine learning": {"ml"},
    "mongodb": {"mongo"},
    "natural language processing": {"nlp"},
    "next js": {"nextjs", "next"},
    "node js": {"nodejs", "node"},
    "postgresql": {"postgres", "psql"},
    "python": {"python3"},
    "react": {"react js", "reactjs"},
    "representational state transfer api": {
        "rest", "rest api", "restful api", "rest apis"
    },
    "structured query language": {"sql"},
    "typescript": {"ts"},
    "large language model": {"llm", "llms", "large language models"},
}

RELATED_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"flask", "django", "fastapi"}),
    frozenset({"react", "angular", "vue"}),
    frozenset({"mysql", "postgresql", "sqlite"}),
    frozenset({"amazon web services", "google cloud platform", "azure"}),
    frozenset({"pytorch", "tensorflow", "keras"}),
)


def normalize_skill(value: str) -> str:
    """Normalize harmless differences while retaining technology identity."""

    value = value.casefold().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def canonical_skill(value: str) -> str:
    normalized = normalize_skill(value)
    for canonical, aliases in ALIAS_GROUPS.items():
        if normalized == canonical or normalized in aliases:
            return canonical
    return normalized


def _are_related(first: str, second: str) -> bool:
    return any(first in group and second in group for group in RELATED_GROUPS)


def _canonical_forms(canonical: str) -> set[str]:
    """Return normalized spellings that can explicitly name a canonical skill."""

    return {canonical, *ALIAS_GROUPS.get(canonical, set())}


def _contains_named_skill(text: str, canonical: str) -> bool:
    """Detect a complete skill phrase inside a longer extracted JD bullet."""

    padded_text = f" {normalize_skill(text)} "
    return any(f" {form} " in padded_text for form in _canonical_forms(canonical))


def collect_resume_skills(skills: Iterable[SkillEvidence]) -> list[SkillEvidence]:
    """Deduplicate evidence by normalized spelling while preserving its first page."""

    collected: dict[str, SkillEvidence] = {}
    for skill in skills:
        collected.setdefault(normalize_skill(skill.name), skill)
    return list(collected.values())


def match_skill(requirement: SkillEvidence, resume_skills: list[SkillEvidence]) -> SkillMatch:
    required_normalized = normalize_skill(requirement.name)
    required_canonical = canonical_skill(requirement.name)

    best: tuple[float, MatchType, SkillEvidence | None] = (0.0, MatchType.MISSING, None)
    for candidate in resume_skills:
        candidate_normalized = normalize_skill(candidate.name)
        candidate_canonical = canonical_skill(candidate.name)

        if required_normalized == candidate_normalized:
            current = (1.0, MatchType.EXACT, candidate)
        elif required_canonical == candidate_canonical:
            current = (0.9, MatchType.EQUIVALENT, candidate)
        elif _contains_named_skill(requirement.name, candidate_canonical):
            # Structured models occasionally return a complete JD bullet instead
            # of one atomic skill. Credit only an explicitly named, bounded skill.
            current = (0.9, MatchType.EQUIVALENT, candidate)
        elif _are_related(required_canonical, candidate_canonical):
            current = (0.5, MatchType.RELATED, candidate)
        else:
            current = (0.0, MatchType.MISSING, None)

        if current[0] > best[0]:
            best = current

    credit, match_type, evidence = best
    return SkillMatch(
        requirement=requirement.name,
        job_page=requirement.source_page,
        match_type=match_type,
        credit=credit,
        resume_skill=evidence.name if evidence else None,
        resume_page=evidence.source_page if evidence else None,
    )


def match_skill_list(
    requirements: list[SkillEvidence],
    resume_skills: list[SkillEvidence],
) -> list[SkillMatch]:
    """Match unique requirements so repeated JD wording cannot distort a score."""

    unique: dict[str, SkillEvidence] = {}
    for requirement in requirements:
        unique.setdefault(canonical_skill(requirement.name), requirement)
    return [match_skill(requirement, resume_skills) for requirement in unique.values()]


def skill_component_score(matches: list[SkillMatch]) -> float:
    if not matches:
        return 100.0
    return round(sum(match.credit for match in matches) / len(matches) * 100, 1)
