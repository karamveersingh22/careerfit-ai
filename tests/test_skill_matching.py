"""Tests for deterministic skill normalization and credit rules."""

from careerfit.models import MatchType, SkillEvidence
from careerfit.skill_matching import (
    canonical_skill,
    match_skill,
    match_skill_list,
    normalize_skill,
    skill_component_score,
)


def skill(name: str, page: int = 1) -> SkillEvidence:
    return SkillEvidence(name=name, source_page=page)


def test_normalization_handles_case_and_punctuation() -> None:
    assert normalize_skill("  Node.JS ") == "node js"
    assert normalize_skill("CI/CD") == "ci cd"


def test_aliases_share_a_canonical_skill() -> None:
    assert canonical_skill("JS") == "javascript"
    assert canonical_skill("JavaScript") == "javascript"
    assert canonical_skill("Postgres") == "postgresql"


def test_exact_match_receives_full_credit() -> None:
    result = match_skill(skill("Python", 2), [skill("python", 1)])

    assert result.match_type == MatchType.EXACT
    assert result.credit == 1.0
    assert result.resume_page == 1


def test_known_alias_receives_equivalent_credit() -> None:
    result = match_skill(skill("JavaScript"), [skill("JS", 3)])

    assert result.match_type == MatchType.EQUIVALENT
    assert result.credit == 0.9


def test_skill_named_inside_compound_job_requirement_is_matched() -> None:
    result = match_skill(
        skill("Proficiency in Python, Java or C++ and familiarity with AI/ML libraries"),
        [skill("Python", 2)],
    )

    assert result.match_type == MatchType.EQUIVALENT
    assert result.resume_skill == "Python"
    assert result.resume_page == 2


def test_alias_named_inside_compound_requirement_is_matched() -> None:
    result = match_skill(
        skill("Exposure to Natural Language Processing (NLP) concepts"),
        [skill("NLP", 3)],
    )

    assert result.match_type == MatchType.EQUIVALENT
    assert result.resume_skill == "NLP"


def test_partial_word_does_not_create_a_false_skill_match() -> None:
    result = match_skill(skill("Communication skills"), [skill("C")])

    assert result.match_type == MatchType.MISSING


def test_related_framework_does_not_receive_full_credit() -> None:
    result = match_skill(skill("Flask"), [skill("Django")])

    assert result.match_type == MatchType.RELATED
    assert result.credit == 0.5


def test_missing_skill_receives_no_credit() -> None:
    result = match_skill(skill("Go"), [skill("Python")])

    assert result.match_type == MatchType.MISSING
    assert result.credit == 0.0
    assert result.resume_skill is None


def test_duplicate_requirements_do_not_distort_component_score() -> None:
    matches = match_skill_list(
        [skill("JavaScript"), skill("JS"), skill("Python")],
        [skill("JS"), skill("Python")],
    )

    assert len(matches) == 2
    assert skill_component_score(matches) == 95.0


def test_absent_job_skill_component_is_not_penalized() -> None:
    assert skill_component_score([]) == 100.0
