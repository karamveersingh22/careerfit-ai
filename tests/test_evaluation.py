"""Regression tests for the balanced Part 6 synthetic benchmark."""

from collections import Counter

from careerfit.evaluation import (
    EvaluationBand,
    build_synthetic_cases,
    run_synthetic_evaluation,
    score_band,
)


def test_benchmark_has_24_balanced_unique_cases() -> None:
    cases = build_synthetic_cases()

    assert len(cases) == 24
    assert len({case.case_id for case in cases}) == 24
    assert Counter(case.expected_band for case in cases) == {
        EvaluationBand.STRONG: 8,
        EvaluationBand.MEDIUM: 8,
        EvaluationBand.WEAK: 8,
    }


def test_all_synthetic_cases_land_in_the_expected_score_band() -> None:
    report = run_synthetic_evaluation()

    assert report.passed_count == 24
    assert report.band_accuracy == 100.0


def test_average_scores_are_meaningfully_separated() -> None:
    report = run_synthetic_evaluation()
    strong = report.average_score(EvaluationBand.STRONG)
    medium = report.average_score(EvaluationBand.MEDIUM)
    weak = report.average_score(EvaluationBand.WEAK)

    assert strong - medium >= 20
    assert medium - weak >= 20


def test_evaluation_is_reproducible() -> None:
    first = run_synthetic_evaluation()
    second = run_synthetic_evaluation()

    assert first == second


def test_score_band_boundaries_are_explicit() -> None:
    assert score_band(75) == EvaluationBand.STRONG
    assert score_band(74.9) == EvaluationBand.MEDIUM
    assert score_band(45) == EvaluationBand.MEDIUM
    assert score_band(44.9) == EvaluationBand.WEAK
