"""Tests for filtered semantic evidence orchestration."""

from careerfit.models import (
    DocumentType,
    EvidenceItem,
    ExtractedDocument,
    ExtractedPage,
    JobProfile,
    ResponsibilityMatch,
    RetrievalHit,
)
from careerfit.rag import build_rag_analysis, merge_responsibility_evidence


class FakeEmbedder:
    def embed_documents(self, texts, titles=None):
        return [[1.0, 0.0] for _ in texts]

    def embed_queries(self, queries):
        return [[1.0, 0.0] for _ in queries]


class FakeStore:
    def __init__(self):
        self.stored = []
        self.queries = []

    def upsert(self, chunks, embeddings):
        self.stored.extend(chunks)

    def query(self, query_embedding, analysis_id, document_type, n_results=3):
        self.queries.append((analysis_id, document_type))
        return [
            RetrievalHit(
                chunk_id="resume-chunk",
                text="Built backend services with Flask.",
                page=1,
                section="experience",
                similarity=0.82,
            )
        ]

    def delete_analysis(self, analysis_id):
        return 0


def doc(kind, text):
    return ExtractedDocument(
        document_type=kind,
        original_filename="test.pdf",
        pages=(ExtractedPage(page_number=1, text=text, character_count=len(text)),),
    )


def test_rag_stores_both_sources_and_filters_queries_to_resume() -> None:
    store = FakeStore()
    result = build_rag_analysis(
        resume_document=doc(DocumentType.RESUME, "EXPERIENCE\nBuilt Flask services"),
        job_document=doc(DocumentType.JOB_DESCRIPTION, "RESPONSIBILITIES\nBuild APIs"),
        job_profile=JobProfile(
            responsibilities=[EvidenceItem(text="Develop backend APIs", source_page=1)]
        ),
        api_key="test",
        analysis_id="analysis-123",
        embedding_client=FakeEmbedder(),
        vector_store=store,
    )

    assert {chunk.document_type for chunk in store.stored} == {
        DocumentType.RESUME,
        DocumentType.JOB_DESCRIPTION,
    }
    assert store.queries == [("analysis-123", DocumentType.RESUME)]
    assert result.semantic_responsibility_matches[0].score == 82.0
    assert result.semantic_responsibility_matches[0].resume_page == 1
    assert result.semantic_responsibility_matches[0].method == "semantic"


def test_low_similarity_is_reported_as_missing_evidence() -> None:
    store = FakeStore()
    store.query = lambda **kwargs: [
        RetrievalHit(
            chunk_id="weak",
            text="Unrelated text",
            page=1,
            section="general",
            similarity=0.2,
        )
    ]
    result = build_rag_analysis(
        doc(DocumentType.RESUME, "Resume text"),
        doc(DocumentType.JOB_DESCRIPTION, "Job text"),
        JobProfile(responsibilities=[EvidenceItem(text="Build APIs", source_page=1)]),
        api_key="test",
        analysis_id="analysis-123",
        embedding_client=FakeEmbedder(),
        vector_store=store,
    )

    match = result.semantic_responsibility_matches[0]
    assert match.score == 0
    assert match.resume_evidence is None


def test_merge_keeps_stronger_evidence() -> None:
    lexical = ResponsibilityMatch(
        requirement="Build APIs", job_page=1, score=75, method="lexical"
    )
    weaker_semantic = ResponsibilityMatch(
        requirement="Build APIs", job_page=1, score=60, method="semantic"
    )
    stronger_semantic = weaker_semantic.model_copy(update={"score": 85})

    assert merge_responsibility_evidence([lexical], [weaker_semantic])[0].method == "lexical"
    assert merge_responsibility_evidence([lexical], [stronger_semantic])[0].method == "semantic"
