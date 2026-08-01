"""Regression tests for the rows rendered by the results dashboard."""

from careerfit.models import MatchType, ResponsibilityMatch, SkillMatch
from careerfit.presentation import build_responsibility_rows, build_skill_rows


def test_skill_rows_do_not_expect_responsibility_metadata() -> None:
    match = SkillMatch(
        requirement="Python",
        job_page=1,
        match_type=MatchType.EXACT,
        credit=1,
        resume_skill="Python",
        resume_page=1,
    )

    rows = build_skill_rows([match])

    assert rows[0]["Requirement"] == "Python"
    assert "Method" not in rows[0]
    assert "Section" not in rows[0]


def test_responsibility_rows_include_retrieval_metadata() -> None:
    match = ResponsibilityMatch(
        requirement="Build APIs",
        job_page=1,
        resume_evidence="Built REST APIs with FastAPI",
        resume_page=2,
        score=88,
        method="semantic",
        section="Experience",
    )

    rows = build_responsibility_rows([match])

    assert rows[0]["Method"] == "semantic"
    assert rows[0]["Section"] == "Experience"
