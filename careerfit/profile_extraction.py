"""Gemini-backed, schema-constrained profile extraction."""

from __future__ import annotations

from typing import TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from .models import (
    DocumentType,
    ExtractedDocument,
    JobProfile,
    ProfileExtractionResult,
    ResumeProfile,
)
from .prompts import build_profile_extraction_prompt

ProfileT = TypeVar("ProfileT", ResumeProfile, JobProfile)


class ProfileExtractionError(RuntimeError):
    """A safe, user-facing extraction failure."""


class GeminiProfileExtractor:
    """Provider adapter for Gemini structured output."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        client=None,
        validation_attempts: int = 2,
    ) -> None:
        if not api_key.strip():
            raise ProfileExtractionError("A Gemini API key has not been configured.")
        if validation_attempts < 1:
            raise ValueError("validation_attempts must be at least one")

        self.model = model
        self.client = client or genai.Client(api_key=api_key)
        self.validation_attempts = validation_attempts

    def extract_pair(
        self,
        resume: ExtractedDocument,
        job_description: ExtractedDocument,
    ) -> ProfileExtractionResult:
        if resume.document_type != DocumentType.RESUME:
            raise ValueError("The first document must be a résumé.")
        if job_description.document_type != DocumentType.JOB_DESCRIPTION:
            raise ValueError("The second document must be a job description.")

        return ProfileExtractionResult(
            model_name=self.model,
            resume_profile=self._extract(resume, ResumeProfile),
            job_profile=self._extract(job_description, JobProfile),
        )

    def _extract(
        self,
        document: ExtractedDocument,
        schema: type[ProfileT],
    ) -> ProfileT:
        prompt = build_profile_extraction_prompt(document)
        last_validation_error: Exception | None = None

        for _ in range(self.validation_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
            except Exception as exc:
                # Do not retry provider, quota, key, or network failures. The
                # project specification permits retries only for invalid output.
                raise ProfileExtractionError(self._safe_provider_error(exc)) from exc

            try:
                output_text = response.text
                if not output_text:
                    raise ValueError("Gemini returned an empty response.")
                profile = schema.model_validate_json(output_text)
                self._validate_page_references(profile, document.page_count)
                return profile
            except (ValidationError, ValueError) as exc:
                last_validation_error = exc

        raise ProfileExtractionError(
            "Gemini returned structured data that failed validation twice. "
            "Please try the extraction again."
        ) from last_validation_error

    def _safe_provider_error(self, exc: Exception) -> str:
        """Turn provider failures into useful messages without exposing secrets."""

        code = getattr(exc, "code", None)
        status_code = getattr(exc, "status_code", None)
        detected_code = code or status_code
        message = str(exc).lower()

        if detected_code in (401, 403) or "permission_denied" in message:
            reason = "The Gemini API key is invalid or lacks model permission."
        elif detected_code == 404 or "not_found" in message:
            reason = f"The configured model `{self.model}` is not available."
        elif detected_code == 429 or "resource_exhausted" in message:
            reason = "The Gemini API quota or rate limit has been reached."
        elif detected_code == 400 or "invalid_request" in message:
            reason = "Gemini rejected the structured extraction request."
        else:
            reason = "Gemini could not be reached or returned a provider error."

        return f"{reason} Active model: `{self.model}`."

    @staticmethod
    def _validate_page_references(profile: BaseModel, page_count: int) -> None:
        """Reject invented page references before data reaches the UI."""

        def visit(value) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "source_page" and not 1 <= child <= page_count:
                        raise ValueError(
                            f"Source page {child} is outside this document."
                        )
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(profile.model_dump())
