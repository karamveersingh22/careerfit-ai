"""Persistent local Chroma storage with mandatory analysis/source filtering."""

from __future__ import annotations

import re
from pathlib import Path

import chromadb

from .models import DocumentChunk, DocumentType, RetrievalHit


class VectorStoreError(RuntimeError):
    """A safe local vector-storage failure."""


class ChromaVectorStore:
    def __init__(
        self,
        path: str | Path,
        embedding_model: str,
        embedding_dimension: int,
        client=None,
    ) -> None:
        safe_model = re.sub(r"[^a-z0-9]+", "-", embedding_model.casefold()).strip("-")
        self.collection_name = f"careerfit-{safe_model}-{embedding_dimension}"
        try:
            self.client = client or chromadb.PersistentClient(path=str(path))
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}},
                metadata={
                    "embedding_model": embedding_model,
                    "embedding_dimension": embedding_dimension,
                },
            )
        except Exception as exc:
            raise VectorStoreError("The local Chroma database could not be opened.") from exc

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")
        if not chunks:
            return
        try:
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                embeddings=embeddings,
                metadatas=[
                    {
                        "analysis_id": chunk.analysis_id,
                        "document_type": chunk.document_type.value,
                        "section": chunk.section,
                        "page": chunk.page,
                    }
                    for chunk in chunks
                ],
            )
        except Exception as exc:
            raise VectorStoreError("Document chunks could not be stored in Chroma.") from exc

    def query(
        self,
        query_embedding: list[float],
        analysis_id: str,
        document_type: DocumentType,
        n_results: int = 3,
    ) -> list[RetrievalHit]:
        """Search only inside one random analysis and one document source."""

        if not analysis_id:
            raise ValueError("analysis_id is required for every vector query")
        if n_results < 1:
            raise ValueError("n_results must be positive")
        where = {
            "$and": [
                {"analysis_id": {"$eq": analysis_id}},
                {"document_type": {"$eq": document_type.value}},
            ]
        }
        try:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError("The filtered Chroma search failed.") from exc

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[RetrievalHit] = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            hits.append(
                RetrievalHit(
                    chunk_id=chunk_id,
                    text=text,
                    page=int(metadata["page"]),
                    section=str(metadata["section"]),
                    similarity=round(similarity, 4),
                )
            )
        return hits

    def count_analysis(self, analysis_id: str) -> int:
        result = self.collection.get(
            where={"analysis_id": {"$eq": analysis_id}},
            include=[],
        )
        return len(result.get("ids") or [])

    def delete_analysis(self, analysis_id: str) -> int:
        """Delete only the records belonging to the requested analysis."""

        count = self.count_analysis(analysis_id)
        if count:
            self.collection.delete(
                where={"analysis_id": {"$eq": analysis_id}}
            )
        return count
