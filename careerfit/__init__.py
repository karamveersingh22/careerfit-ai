"""CareerFit AI application package."""

from .document_engine import extract_pdf, validate_pdf_upload
from .models import (
    DocumentType,
    ExtractedDocument,
    ExtractedPage,
    JobProfile,
    ProfileExtractionResult,
    RagAnalysisResult,
    ResumeProfile,
    ScoringResult,
)
from .profile_extraction import GeminiProfileExtractor, ProfileExtractionError
from .scoring import score_profiles
from .rag import RagError, build_rag_analysis

__all__ = [
    "DocumentType",
    "ExtractedDocument",
    "ExtractedPage",
    "GeminiProfileExtractor",
    "JobProfile",
    "ProfileExtractionError",
    "ProfileExtractionResult",
    "RagAnalysisResult",
    "RagError",
    "ResumeProfile",
    "ScoringResult",
    "extract_pdf",
    "validate_pdf_upload",
    "score_profiles",
    "build_rag_analysis",
]
