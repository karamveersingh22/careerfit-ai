"""Integration tests for local persistent Chroma filtering and deletion."""

from careerfit.models import DocumentChunk, DocumentType
from careerfit.vector_store import ChromaVectorStore


def chunk(chunk_id, analysis_id, document_type, text, page=1):
    return DocumentChunk(
        chunk_id=chunk_id,
        analysis_id=analysis_id,
        document_type=document_type,
        section="experience",
        page=page,
        text=text,
    )


def test_query_isolated_by_analysis_and_document_type(tmp_path) -> None:
    store = ChromaVectorStore(tmp_path, "test-model", 2)
    chunks = [
        chunk("a-resume", "analysis-a", DocumentType.RESUME, "Python APIs"),
        chunk("a-job", "analysis-a", DocumentType.JOB_DESCRIPTION, "Python required"),
        chunk("b-resume", "analysis-b", DocumentType.RESUME, "Different candidate"),
    ]
    store.upsert(chunks, [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

    hits = store.query(
        [1.0, 0.0], "analysis-a", DocumentType.RESUME, n_results=3
    )

    assert [hit.chunk_id for hit in hits] == ["a-resume"]
    assert hits[0].page == 1
    assert hits[0].similarity == 1.0


def test_delete_removes_only_one_analysis(tmp_path) -> None:
    store = ChromaVectorStore(tmp_path, "test-model", 2)
    chunks = [
        chunk("a", "analysis-a", DocumentType.RESUME, "A"),
        chunk("b", "analysis-b", DocumentType.RESUME, "B"),
    ]
    store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]])

    assert store.delete_analysis("analysis-a") == 1
    assert store.count_analysis("analysis-a") == 0
    assert store.count_analysis("analysis-b") == 1


def test_upsert_requires_one_vector_per_chunk(tmp_path) -> None:
    store = ChromaVectorStore(tmp_path, "test-model", 2)

    try:
        store.upsert(
            [chunk("a", "analysis-a", DocumentType.RESUME, "A")], []
        )
    except ValueError as exc:
        assert "exactly one embedding" in str(exc)
    else:
        raise AssertionError("Expected a ValueError")
