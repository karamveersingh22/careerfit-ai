"""Part 4 indexing and page-linked semantic evidence retrieval."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from .chunking import chunk_document
from .embeddings import EmbeddingError, GeminiEmbeddingClient
from .models import (
    DocumentType,
    ExtractedDocument,
    JobProfile,
    RagAnalysisResult,
    ResponsibilityMatch,
)
from .vector_store import ChromaVectorStore, VectorStoreError

MINIMUM_EVIDENCE_SIMILARITY = 0.45


class RagError(RuntimeError):
    """A safe end-to-end retrieval failure."""


def build_rag_analysis(
    resume_document: ExtractedDocument,
    job_document: ExtractedDocument,
    job_profile: JobProfile,
    api_key: str,
    vector_path: str | Path = "data/chroma",
    embedding_model: str = "gemini-embedding-2",
    embedding_dimension: int = 768,
    analysis_id: str | None = None,
    embedding_client: GeminiEmbeddingClient | None = None,
    vector_store: ChromaVectorStore | None = None,
) -> RagAnalysisResult:
    """Embed, persist, and retrieve responsibility evidence for one analysis."""

    analysis_id = analysis_id or str(uuid4())
    embedder = embedding_client or GeminiEmbeddingClient(
        api_key=api_key,
        model=embedding_model,
        dimension=embedding_dimension,
    )
    store = vector_store or ChromaVectorStore(
        path=vector_path,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
    )

    try:
        chunks = chunk_document(resume_document, analysis_id) + chunk_document(
            job_document, analysis_id
        )
        document_vectors = embedder.embed_documents(
            [chunk.text for chunk in chunks],
            [f"{chunk.document_type.value} {chunk.section}" for chunk in chunks],
        )
        store.upsert(chunks, document_vectors)

        requirements = job_profile.responsibilities
        query_vectors = embedder.embed_queries(
            [requirement.text for requirement in requirements]
        )
        semantic_matches: list[ResponsibilityMatch] = []
        for requirement, query_vector in zip(
            requirements, query_vectors, strict=True
        ):
            hits = store.query(
                query_embedding=query_vector,
                analysis_id=analysis_id,
                document_type=DocumentType.RESUME,
                n_results=3,
            )
            best = hits[0] if hits else None
            has_evidence = bool(
                best and best.similarity >= MINIMUM_EVIDENCE_SIMILARITY
            )
            semantic_matches.append(
                ResponsibilityMatch(
                    requirement=requirement.text,
                    job_page=requirement.source_page,
                    resume_evidence=best.text if has_evidence else None,
                    resume_page=best.page if has_evidence else None,
                    score=round(best.similarity * 100, 1) if has_evidence else 0.0,
                    method="semantic",
                    section=best.section if has_evidence else None,
                    chunk_id=best.chunk_id if has_evidence else None,
                )
            )
    except (EmbeddingError, VectorStoreError, ValueError) as exc:
        # Avoid leaving a partial analysis behind if indexing or retrieval fails.
        try:
            store.delete_analysis(analysis_id)
        except Exception:
            pass
        raise RagError(str(exc)) from exc

    return RagAnalysisResult(
        analysis_id=analysis_id,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        chunk_count=len(chunks),
        semantic_responsibility_matches=semantic_matches,
    )


def merge_responsibility_evidence(
    lexical: list[ResponsibilityMatch],
    semantic: list[ResponsibilityMatch],
) -> list[ResponsibilityMatch]:
    """Keep the strongest transparent evidence for each requirement."""

    semantic_by_key = {
        (match.requirement, match.job_page): match for match in semantic
    }
    merged: list[ResponsibilityMatch] = []
    for lexical_match in lexical:
        semantic_match = semantic_by_key.get(
            (lexical_match.requirement, lexical_match.job_page)
        )
        if semantic_match and semantic_match.score > lexical_match.score:
            merged.append(semantic_match)
        else:
            merged.append(lexical_match)
    return merged
