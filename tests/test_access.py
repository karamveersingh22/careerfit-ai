"""Tests for shared-demo and visitor-key access isolation."""

import pytest

from careerfit.access import (
    AccessConfigurationError,
    AccessMode,
    remaining_demo_analyses,
    resolve_gemini_key,
)


def test_demo_has_three_session_analyses() -> None:
    assert remaining_demo_analyses(0) == 3
    assert remaining_demo_analyses(2) == 1
    assert remaining_demo_analyses(3) == 0


def test_demo_uses_only_shared_key() -> None:
    assert resolve_gemini_key(
        AccessMode.SHARED_DEMO,
        shared_key="shared",
        visitor_key="visitor",
        demo_analyses_used=0,
    ) == "shared"


def test_exhausted_demo_does_not_fall_back_to_visitor_key() -> None:
    with pytest.raises(AccessConfigurationError, match="all three"):
        resolve_gemini_key(
            AccessMode.SHARED_DEMO,
            shared_key="shared",
            visitor_key="visitor",
            demo_analyses_used=3,
        )


def test_visitor_mode_does_not_fall_back_to_shared_key() -> None:
    with pytest.raises(AccessConfigurationError, match="Paste your Gemini"):
        resolve_gemini_key(
            AccessMode.VISITOR_KEY,
            shared_key="shared",
            visitor_key="",
            demo_analyses_used=0,
        )


def test_visitor_mode_ignores_demo_limit() -> None:
    assert resolve_gemini_key(
        AccessMode.VISITOR_KEY,
        shared_key="shared",
        visitor_key="visitor",
        demo_analyses_used=3,
    ) == "visitor"
