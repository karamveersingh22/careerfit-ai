import os
from typing import List

import chromadb
import google.generativeai as genai
import streamlit as st

from careerfit.matching import EvidenceMatch, ResumeJobMatcher


class GeminiEmbeddingModel:
    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            result = genai.embed_content(model="models/text-embedding-004", content=text)
            vectors.append(result["embedding"])
        return vectors


class GeminiExplanationModel:
    def explain(self, job_description: str, evidence: List[EvidenceMatch]) -> str:
        model = genai.GenerativeModel("gemini-1.5-flash")
        evidence_text = "\n".join(
            f"- Score {item.score:.2f}: {item.text}" for item in evidence
        )
        prompt = (
            "You are evaluating resume fit for a job. Use only evidence listed below. "
            "Give a short match summary and mention missing skills if any.\n\n"
            f"Job Description:\n{job_description}\n\n"
            f"Evidence:\n{evidence_text}"
        )
        return model.generate_content(prompt).text


class RuleBasedExplanationModel:
    def explain(self, job_description: str, evidence: List[EvidenceMatch]) -> str:
        avg_score = sum(item.score for item in evidence) / len(evidence)
        top_evidence = "\n".join(f"- {item.text}" for item in evidence[:3])
        return (
            f"Average evidence score: {avg_score:.2f}. "
            "Set GOOGLE_API_KEY to enable Gemini-generated narrative.\n"
            f"Top evidence:\n{top_evidence}"
        )


def create_matcher() -> ResumeJobMatcher:
    collection = chromadb.Client().create_collection(name="resume_matching")
    embedding_model = GeminiEmbeddingModel()
    if os.environ.get("GOOGLE_API_KEY"):
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        explanation_model = GeminiExplanationModel()
    else:
        explanation_model = RuleBasedExplanationModel()
    return ResumeJobMatcher(collection, embedding_model, explanation_model)


def main() -> None:
    st.set_page_config(page_title="CareerFit AI", page_icon="📄")
    st.title("CareerFit AI")
    st.caption("Evidence-grounded resume and job-description matching")

    resume_text = st.text_area("Resume text", height=220)
    job_description = st.text_area("Job description", height=220)
    top_k = st.slider("Evidence chunks", min_value=1, max_value=10, value=5)

    if st.button("Match"):
        if not resume_text.strip() or not job_description.strip():
            st.warning("Please provide both resume and job description text.")
            return

        matcher = create_matcher()
        evidence, explanation = matcher.match(resume_text, job_description, top_k=top_k)

        st.subheader("Match explanation")
        st.write(explanation)

        st.subheader("Evidence")
        for item in evidence:
            st.markdown(f"**Score: {item.score:.2f}**")
            st.write(item.text)


if __name__ == "__main__":
    main()
