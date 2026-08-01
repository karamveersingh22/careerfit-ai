"""Tests for Gemini embedding formatting and vector validation."""

from types import SimpleNamespace

import pytest

from careerfit.embeddings import EmbeddingError, GeminiEmbeddingClient


class FakeModels:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=vector) for vector in self.vectors]
        )


def test_document_and_query_formatting_and_normalization() -> None:
    models = FakeModels([[3.0, 4.0], [0.0, 2.0]])
    client = GeminiEmbeddingClient(
        api_key="test-key",
        model="test-embedding",
        dimension=2,
        client=SimpleNamespace(models=models),
    )

    document_vectors = client.embed_documents(
        ["Python APIs", "SQL data"], ["experience", "skills"]
    )

    assert document_vectors[0] == pytest.approx([0.6, 0.8])
    assert document_vectors[1] == pytest.approx([0.0, 1.0])
    contents = models.calls[0]["contents"]
    assert contents[0].parts[0].text.startswith("title: experience | text:")

    models.vectors = [[1.0, 0.0]]
    client.embed_queries(["Build APIs"])
    query_content = models.calls[1]["contents"][0].parts[0].text
    assert query_content.startswith("task: search result | query:")


def test_rejects_wrong_embedding_count_or_dimension() -> None:
    missing = GeminiEmbeddingClient(
        api_key="test-key",
        dimension=2,
        client=SimpleNamespace(models=FakeModels([])),
    )
    with pytest.raises(EmbeddingError, match="unexpected number"):
        missing.embed_queries(["query"])

    wrong_dimension = GeminiEmbeddingClient(
        api_key="test-key",
        dimension=2,
        client=SimpleNamespace(models=FakeModels([[1.0, 2.0, 3.0]])),
    )
    with pytest.raises(EmbeddingError, match="dimension"):
        wrong_dimension.embed_queries(["query"])


def test_rejects_zero_vector() -> None:
    client = GeminiEmbeddingClient(
        api_key="test-key",
        dimension=2,
        client=SimpleNamespace(models=FakeModels([[0.0, 0.0]])),
    )
    with pytest.raises(EmbeddingError, match="empty embedding"):
        client.embed_queries(["query"])
