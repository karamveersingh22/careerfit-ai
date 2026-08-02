"""Session-scoped access rules for shared demo and visitor credentials."""

from __future__ import annotations

from enum import StrEnum


DEMO_ANALYSIS_LIMIT = 3


class AccessMode(StrEnum):
    """Supported ways to authorize Gemini requests."""

    SHARED_DEMO = "Shared demo"
    VISITOR_KEY = "Use my Gemini key"


class AccessConfigurationError(ValueError):
    """A safe error for missing or exhausted credentials."""


def remaining_demo_analyses(
    analyses_used: int, limit: int = DEMO_ANALYSIS_LIMIT
) -> int:
    """Return the non-negative number of demo analyses left this session."""

    return max(0, limit - max(0, analyses_used))


def resolve_gemini_key(
    mode: AccessMode,
    *,
    shared_key: str | None,
    visitor_key: str | None,
    demo_analyses_used: int,
) -> str:
    """Choose one credential without ever falling back between access modes."""

    if mode == AccessMode.VISITOR_KEY:
        key = (visitor_key or "").strip()
        if not key:
            raise AccessConfigurationError(
                "Paste your Gemini API key before analyzing the documents."
            )
        return key

    if remaining_demo_analyses(demo_analyses_used) == 0:
        raise AccessConfigurationError(
            "You have used all three shared demo analyses in this session. "
            "Choose 'Use my Gemini key' to continue."
        )

    key = (shared_key or "").strip()
    if not key:
        raise AccessConfigurationError(
            "The shared demo is temporarily unavailable. You can still use "
            "your own Gemini API key."
        )
    return key
