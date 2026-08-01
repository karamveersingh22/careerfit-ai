"""Gemini Embedding 2 adapter for asymmetric document retrieval."""

from __future__ import annotations

import math

from google import genai
from google.genai import types


class EmbeddingError(RuntimeError):
    """A safe embedding provider failure."""


class GeminiEmbeddingClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-2",
        dimension: int = 768,
        client=None,
    ) -> None:
        if not api_key.strip():
            raise EmbeddingError("A Gemini API key has not been configured.")
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")
        self.model = model
        self.dimension = dimension
        self.client = client or genai.Client(api_key=api_key)

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            raise EmbeddingError("Gemini returned an empty embedding vector.")
        return [value / magnitude for value in vector]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            contents = [
                types.Content(parts=[types.Part.from_text(text=text)])
                for text in texts
            ]
            response = self.client.models.embed_content(
                model=self.model,
                contents=contents,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dimension
                ),
            )
            vectors = [list(item.values) for item in response.embeddings]
        except Exception as exc:
            raise EmbeddingError(
                f"Gemini embeddings failed. Active model: `{self.model}`."
            ) from exc

        if len(vectors) != len(texts):
            raise EmbeddingError("Gemini returned an unexpected number of embeddings.")
        if any(len(vector) != self.dimension for vector in vectors):
            raise EmbeddingError(
                f"Gemini returned a vector dimension different from {self.dimension}."
            )
        return [self._normalize(vector) for vector in vectors]

    def embed_documents(
        self, texts: list[str], titles: list[str] | None = None
    ) -> list[list[float]]:
        if titles is not None and len(titles) != len(texts):
            raise ValueError("titles must have the same length as texts")
        prepared = [
            f"title: {(titles[index] if titles else 'none')} | text: {text}"
            for index, text in enumerate(texts)
        ]
        return self._embed(prepared)

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        prepared = [f"task: search result | query: {query}" for query in queries]
        return self._embed(prepared)
