from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


class EmbeddingModel(Protocol):
    def embed(self, texts: List[str]) -> List[List[float]]: ...


class ExplanationModel(Protocol):
    def explain(self, job_description: str, evidence: List["EvidenceMatch"]) -> str: ...


@dataclass
class EvidenceMatch:
    text: str
    score: float


def chunk_text(text: str, chunk_size: int = 400) -> List[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    words = cleaned.split(" ")
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > chunk_size:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        chunks.append(" ".join(current))
    return chunks


class ResumeJobMatcher:
    def __init__(self, collection, embedding_model: EmbeddingModel, explanation_model: ExplanationModel):
        self.collection = collection
        self.embedding_model = embedding_model
        self.explanation_model = explanation_model

    def find_evidence(self, resume_text: str, job_description: str, top_k: int = 5) -> List[EvidenceMatch]:
        chunks = chunk_text(resume_text)
        if not chunks:
            return []

        ids = [f"resume-chunk-{i}" for i in range(len(chunks))]
        embeddings = self.embedding_model.embed(chunks)

        self.collection.add(ids=ids, documents=chunks, embeddings=embeddings)

        query_embedding = self.embedding_model.embed([job_description])[0]
        query_result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(chunks)),
            include=["documents", "distances"],
        )

        docs = query_result["documents"][0]
        distances = query_result.get("distances", [[0.0] * len(docs)])[0]

        evidence: List[EvidenceMatch] = []
        for doc, distance in zip(docs, distances):
            score = 1.0 / (1.0 + float(distance))
            evidence.append(EvidenceMatch(text=doc, score=score))
        return evidence

    def match(self, resume_text: str, job_description: str, top_k: int = 5) -> tuple[List[EvidenceMatch], str]:
        evidence = self.find_evidence(resume_text, job_description, top_k=top_k)
        if not evidence:
            return [], "No resume evidence could be extracted."
        explanation = self.explanation_model.explain(job_description, evidence)
        return evidence, explanation
