"""Convert domain models into rows that the Streamlit UI can display."""

from __future__ import annotations

from collections.abc import Iterable

from careerfit.models import ResponsibilityMatch, SkillMatch


def build_skill_rows(matches: Iterable[SkillMatch]) -> list[dict[str, object]]:
    """Build rows using only fields that belong to a skill match."""

    return [
        {
            "Requirement": match.requirement,
            "Status": match.match_type.value,
            "Credit": match.credit,
            "Résumé evidence": match.resume_skill or "No evidence found",
            "Résumé page": match.resume_page,
            "Job page": match.job_page,
        }
        for match in matches
    ]


def build_responsibility_rows(
    matches: Iterable[ResponsibilityMatch],
) -> list[dict[str, object]]:
    """Build evidence rows, including the Part 4 retrieval metadata."""

    return [
        {
            "Job responsibility": match.requirement,
            "Match score": match.score,
            "Résumé evidence": match.resume_evidence or "No evidence found",
            "Résumé page": match.resume_page,
            "Job page": match.job_page,
            "Method": match.method,
            "Section": match.section,
        }
        for match in matches
    ]
