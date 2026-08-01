"""Streamlit entry point for CareerFit AI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from careerfit.document_engine import PDFValidationError, extract_pdf
from careerfit.evaluation import EvaluationBand, run_synthetic_evaluation
from careerfit.guidance import GeminiGuidanceGenerator, GuidanceError
from careerfit.models import DocumentType, ExtractedDocument
from careerfit.profile_extraction import GeminiProfileExtractor, ProfileExtractionError
from careerfit.presentation import build_responsibility_rows, build_skill_rows
from careerfit.embeddings import GeminiEmbeddingClient
from careerfit.rag import RagError, build_rag_analysis, merge_responsibility_evidence
from careerfit.scoring import score_profiles
from careerfit.vector_store import ChromaVectorStore, VectorStoreError


st.set_page_config(
    page_title="CareerFit AI",
    page_icon=":material/target:",
    layout="wide",
)

st.session_state.setdefault("resume_document", None)
st.session_state.setdefault("jd_document", None)
st.session_state.setdefault("profile_result", None)
st.session_state.setdefault("score_result", None)
st.session_state.setdefault("rag_result", None)
st.session_state.setdefault("rag_error", None)
st.session_state.setdefault("guidance_result", None)
st.session_state.setdefault("guidance_error", None)


def read_setting(name: str, default: Any = None) -> Any:
    """Read a backend setting without exposing it in the browser."""

    environment_value = os.getenv(name)
    if environment_value:
        return environment_value
    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:
        return default


@st.cache_resource
def get_profile_extractor(api_key: str, model: str) -> GeminiProfileExtractor:
    """Reuse the provider client while keeping it outside per-user state."""

    return GeminiProfileExtractor(api_key=api_key, model=model)


@st.cache_resource
def get_guidance_generator(api_key: str, model: str) -> GeminiGuidanceGenerator:
    """Reuse the guidance provider client across Streamlit reruns."""

    return GeminiGuidanceGenerator(api_key=api_key, model=model)


@st.cache_resource
def get_embedding_client(
    api_key: str, model: str, dimension: int
) -> GeminiEmbeddingClient:
    return GeminiEmbeddingClient(
        api_key=api_key, model=model, dimension=dimension
    )


@st.cache_resource
def get_vector_store(model: str, dimension: int) -> ChromaVectorStore:
    return ChromaVectorStore(
        path=Path("data/chroma"),
        embedding_model=model,
        embedding_dimension=dimension,
    )


@st.cache_data
def get_evaluation_report():
    """Cache the deterministic Part 6 benchmark across UI reruns."""

    return run_synthetic_evaluation()


def clear_analysis_outputs() -> None:
    for key in (
        "resume_document",
        "jd_document",
        "profile_result",
        "score_result",
        "rag_result",
        "rag_error",
        "guidance_result",
        "guidance_error",
    ):
        st.session_state[key] = None


def render_document_summary(document: ExtractedDocument, label: str) -> None:
    """Show safe extraction metadata and an optional page-aware preview."""

    st.success(f"{label} extracted successfully")
    first, second = st.columns(2)
    first.metric("Pages", document.page_count)
    second.metric("Readable characters", f"{document.character_count:,}")

    with st.expander(f"Preview extracted {label.lower()} text"):
        st.caption(
            "Page markers are preserved so later scores and recommendations can "
            "cite their evidence."
        )
        for page in document.pages:
            st.markdown(f"**Page {page.page_number}**")
            st.text(page.text or "No readable text on this page.")


def process_upload(uploaded_file, document_type: DocumentType):
    """Extract one Streamlit upload without saving it to disk."""

    if uploaded_file is None:
        return None
    return extract_pdf(uploaded_file.name, uploaded_file.getvalue(), document_type)


st.title("CareerFit AI")
st.write(
    "Upload a résumé and a target job description. CareerFit validates both "
    "PDFs, preserves page evidence, and extracts validated structured profiles."
)

api_key = read_setting("GEMINI_API_KEY")
model_name = (
    read_setting("GEMINI_MODEL", "gemini-3.5-flash-lite")
    or "gemini-3.5-flash-lite"
)
embedding_model = (
    read_setting("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    or "gemini-embedding-2"
)
embedding_dimension = int(read_setting("GEMINI_EMBEDDING_DIMENSION", 768))

with st.sidebar:
    st.header("Parts 1–6")
    st.info(
        "Only text-based PDFs are supported for now. Files are processed in "
        "memory and are not saved by this application."
    )
    st.caption("Maximum file size: 10 MB · Maximum pages: 50 per PDF")
    if api_key:
        st.success("Gemini is configured", icon=":material/check_circle:")
        st.caption(f"Structured extraction model: `{model_name}`")
        st.caption(
            f"Embedding model: `{embedding_model}` ({embedding_dimension} dimensions)"
        )
    else:
        st.warning("Gemini API key is not configured", icon=":material/key:")
        st.caption("Add `GEMINI_API_KEY` to `.streamlit/secrets.toml`.")
    st.warning(
        "Gemini free-tier requests may be used by Google to improve its products. "
        "Use synthetic documents while developing and review provider terms before "
        "submitting a real résumé.",
        icon=":material/privacy_tip:",
    )
    with st.expander("Part 6 quality benchmark"):
        benchmark = get_evaluation_report()
        st.metric(
            "Expected score bands",
            f"{benchmark.passed_count}/{benchmark.case_count}",
        )
        st.caption(
            f"{benchmark.band_accuracy:.1f}% accuracy across eight synthetic "
            "job families, with eight strong, eight medium, and eight weak cases."
        )
        st.dataframe(
            [
                {
                    "Band": band.value,
                    "Average score": benchmark.average_score(band),
                }
                for band in EvaluationBand
            ],
            hide_index=True,
            column_config={
                "Average score": st.column_config.ProgressColumn(
                    "Average score", min_value=0, max_value=100, format="%.1f%%"
                )
            },
        )

with st.form("document_analysis", border=False):
    resume_column, jd_column = st.columns(2, gap="large")
    with resume_column:
        st.subheader("1. Résumé")
        resume_upload = st.file_uploader(
            "Choose a résumé PDF",
            type=["pdf"],
            key="resume_upload",
        )

    with jd_column:
        st.subheader("2. Job description")
        jd_upload = st.file_uploader(
            "Choose a job-description PDF",
            type=["pdf"],
            key="jd_upload",
        )

    provider_consent = st.checkbox(
        "I understand that extracted document text will be sent to the configured "
        "Gemini API for profile extraction and grounded career guidance.",
        key="provider_consent",
    )

    submitted = st.form_submit_button(
        "Analyze documents",
        type="primary",
        icon=":material/document_search:",
        width="stretch",
    )

if submitted:
    if not resume_upload or not jd_upload:
        st.warning("Upload both documents to continue.")
    elif not provider_consent:
        st.warning("Confirm the Gemini data-transfer notice before continuing.")
    elif not api_key:
        st.error(
            "Gemini is not configured. Add your API key to "
            "`.streamlit/secrets.toml`, then restart the app."
        )
    else:
        try:
            with st.spinner("Extracting profiles and building the evidence index..."):
                resume_document = process_upload(
                    resume_upload, DocumentType.RESUME
                )
                jd_document = process_upload(
                    jd_upload, DocumentType.JOB_DESCRIPTION
                )
                extractor = get_profile_extractor(api_key, model_name)
                profile_result = extractor.extract_pair(
                    resume_document, jd_document
                )
                score_result = score_profiles(profile_result, resume_document)
                previous_rag = st.session_state.get("rag_result")
                if previous_rag:
                    try:
                        get_vector_store(
                            previous_rag.embedding_model,
                            previous_rag.embedding_dimension,
                        ).delete_analysis(previous_rag.analysis_id)
                    except VectorStoreError as exc:
                        raise RagError(
                            "The previous local analysis could not be removed."
                        ) from exc
                rag_result = None
                rag_error = None
                guidance_result = None
                guidance_error = None
                try:
                    rag_result = build_rag_analysis(
                        resume_document=resume_document,
                        job_document=jd_document,
                        job_profile=profile_result.job_profile,
                        api_key=api_key,
                        embedding_model=embedding_model,
                        embedding_dimension=embedding_dimension,
                        embedding_client=get_embedding_client(
                            api_key, embedding_model, embedding_dimension
                        ),
                        vector_store=get_vector_store(
                            embedding_model, embedding_dimension
                        ),
                    )
                    combined_evidence = merge_responsibility_evidence(
                        score_result.responsibility_matches,
                        rag_result.semantic_responsibility_matches,
                    )
                    score_result = score_profiles(
                        profile_result,
                        resume_document,
                        responsibility_evidence_override=combined_evidence,
                    )
                except RagError as exc:
                    rag_error = str(exc)
                try:
                    guidance_result = get_guidance_generator(
                        api_key, model_name
                    ).generate(
                        profile_result,
                        score_result,
                        resume_document,
                        jd_document,
                    )
                except GuidanceError as exc:
                    guidance_error = str(exc)
                st.session_state["resume_document"] = resume_document
                st.session_state["jd_document"] = jd_document
                st.session_state["profile_result"] = profile_result
                st.session_state["score_result"] = score_result
                st.session_state["rag_result"] = rag_result
                st.session_state["rag_error"] = rag_error
                st.session_state["guidance_result"] = guidance_result
                st.session_state["guidance_error"] = guidance_error
        except (PDFValidationError, ProfileExtractionError, RagError) as exc:
            st.error(str(exc))

if st.session_state["resume_document"] and st.session_state["jd_document"]:
    st.divider()
    st.header("Extraction results")
    resume_result, jd_result = st.columns(2, gap="large")
    with resume_result:
        render_document_summary(st.session_state["resume_document"], "Résumé")
    with jd_result:
        render_document_summary(st.session_state["jd_document"], "Job description")

if st.session_state["profile_result"]:
    result = st.session_state["profile_result"]
    st.divider()
    st.header("Validated profiles")
    st.caption(
        f"Extracted with `{result.model_name}`. Unknown fields remain null; "
        "every evidence item retains its source page."
    )
    resume_profile_tab, job_profile_tab = st.tabs(
        ["Résumé profile", "Job profile"]
    )
    with resume_profile_tab:
        st.json(result.resume_profile.model_dump(mode="json"), expanded=1)
    with job_profile_tab:
        st.json(result.job_profile.model_dump(mode="json"), expanded=1)
else:
    st.caption(
        "Your documents remain local until you submit both files. Only the "
        "extracted text is sent to Gemini for structured profile extraction."
    )

if (
    st.session_state["profile_result"]
    and st.session_state["resume_document"]
    and not st.session_state["score_result"]
):
    st.session_state["score_result"] = score_profiles(
        st.session_state["profile_result"],
        st.session_state["resume_document"],
    )

if st.session_state["score_result"]:
    score = st.session_state["score_result"]
    st.divider()
    st.header("Transparent scoring")
    st.caption(
        "These scores are calculated by deterministic Python rules. Gemini "
        "extracts evidence but does not choose any score."
    )

    with st.container(horizontal=True):
        st.metric(
            "CareerFit match",
            f"{score.overall_match_score:.1f}%",
            border=True,
        )
        st.metric(
            "Content readiness",
            f"{score.ats_readiness_score:.1f}%",
            border=True,
        )
        st.metric(
            "Matched skills",
            len(score.matched_skills),
            border=True,
        )
        st.metric(
            "Missing skills",
            len(score.missing_skills),
            border=True,
        )

    component_labels = {
        "required_skills": "Required skills",
        "experience": "Experience",
        "responsibilities": "Responsibilities",
        "preferred_skills": "Preferred skills",
        "education": "Education",
        "resume_quality": "Résumé quality",
    }
    component_rows = [
        {
            "Component": component_labels[name],
            "Score": component_score,
            "Weight": score.component_weights[name],
            "Contribution": round(
                component_score * score.component_weights[name], 1
            ),
        }
        for name, component_score in score.component_scores.items()
    ]

    component_tab, skills_tab, readiness_tab, evidence_tab = st.tabs(
        ["Formula", "Skill matches", "Readiness", "Responsibility evidence"]
    )
    with component_tab:
        st.dataframe(
            component_rows,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=100, format="%.1f%%"
                ),
                "Weight": st.column_config.NumberColumn(
                    "Weight", format="percent"
                ),
                "Contribution": st.column_config.NumberColumn(
                    "Contribution", format="%.1f points"
                ),
            },
        )
        st.caption(
            "Overall match = the sum of each component score multiplied by its "
            "published weight."
        )

    with skills_tab:
        skill_rows = build_skill_rows(score.skill_matches)
        if skill_rows:
            st.dataframe(
                skill_rows,
                hide_index=True,
                column_config={
                    "Credit": st.column_config.ProgressColumn(
                        "Credit", min_value=0, max_value=1, format="percent"
                    )
                },
            )
            st.caption(
                "Credit rules: exact 100%, known equivalent 90%, deliberately "
                "related technology 50%, missing evidence 0%."
            )
        else:
            st.info("The job description contained no explicit skill requirements.")

    with readiness_tab:
        readiness_rows = [
            {
                "Check": check.name,
                "Score": check.score,
                "Weight": check.weight,
                "Explanation": check.explanation,
            }
            for check in score.readiness_checks
        ]
        st.dataframe(
            readiness_rows,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=100, format="%.1f%%"
                ),
                "Weight": st.column_config.NumberColumn(
                    "Weight", format="percent"
                ),
            },
        )

    with evidence_tab:
        evidence_rows = build_responsibility_rows(score.responsibility_matches)
        if evidence_rows:
            st.dataframe(
                evidence_rows,
                hide_index=True,
                column_config={
                    "Match score": st.column_config.ProgressColumn(
                        "Match score", min_value=0, max_value=100, format="%.1f%%"
                    )
                },
            )
        else:
            st.info("The job description contained no explicit responsibilities.")

    with st.expander("Formula limitations"):
        for limitation in score.limitations:
            st.markdown(f"- {limitation}")

if st.session_state["guidance_error"]:
    st.warning(
        "Career guidance could not be generated, but your scores and evidence "
        f"are unaffected. {st.session_state['guidance_error']}"
    )

if st.session_state["guidance_result"]:
    guidance = st.session_state["guidance_result"]
    st.divider()
    st.header("Grounded career guidance")
    st.caption(
        f"Generated with `{guidance.model_name}` from the finished analysis. "
        "It does not change any score, and every factual recommendation points "
        "back to the uploaded documents."
    )

    strengths_tab, gaps_tab, rewrites_tab, learning_tab, interviews_tab = st.tabs(
        ["Strengths", "Gaps", "Résumé rewrites", "Learning plan", "Interviews"]
    )

    with strengths_tab:
        if not guidance.strengths:
            st.info("No sufficiently grounded strengths were generated.")
        for item in guidance.strengths:
            with st.container(border=True):
                st.subheader(item.title)
                st.write(item.explanation)
                st.markdown(f"**Résumé evidence:** {item.resume_evidence}")
                st.caption(
                    f"Résumé page {item.resume_page} · Job page {item.job_page} · "
                    f"Target evidence: {item.job_evidence}"
                )

    with gaps_tab:
        if not guidance.gaps:
            st.success("No material evidence gaps were identified.")
        for item in guidance.gaps:
            with st.container(border=True):
                st.subheader(item.title)
                st.write(item.explanation)
                if item.resume_evidence:
                    st.markdown(f"**Closest résumé evidence:** {item.resume_evidence}")
                    resume_source = f"Résumé page {item.resume_page} · "
                else:
                    st.markdown("**Résumé evidence:** No explicit evidence found")
                    resume_source = ""
                st.caption(
                    f"{resume_source}Job page {item.job_page} · "
                    f"Target evidence: {item.job_evidence}"
                )

    with rewrites_tab:
        st.info(
            "These suggestions only rephrase existing evidence. Review and edit "
            "them before using them in your résumé."
        )
        if not guidance.rewrites:
            st.info("No safe evidence-based rewrites were generated.")
        for item in guidance.rewrites:
            with st.container(border=True):
                st.markdown(f"**Original:** {item.original_text}")
                st.markdown(f"**Suggested rewrite:** {item.improved_text}")
                st.write(item.reason)
                st.caption(
                    f"Résumé page {item.resume_page} · Job page {item.job_page} · "
                    f"Target evidence: {item.target_job_evidence}"
                )

    with learning_tab:
        if not guidance.learning_plan:
            st.info("No job-specific learning actions were generated.")
        for index, item in enumerate(guidance.learning_plan, start=1):
            with st.container(border=True):
                st.subheader(f"{index}. {item.topic}")
                st.write(item.why_it_matters)
                st.markdown(f"**Suggested action:** {item.suggested_action}")
                st.caption(
                    f"Job page {item.job_page} · Target evidence: {item.job_evidence}"
                )

    with interviews_tab:
        if not guidance.interview_questions:
            st.info("No grounded interview questions were generated.")
        for item in guidance.interview_questions:
            with st.container(border=True):
                st.badge(item.category.capitalize())
                st.subheader(item.question)
                st.markdown(f"**Preparation tip:** {item.preparation_tip}")
                sources = f"Job page {item.job_page}"
                if item.resume_page:
                    sources += f" · Résumé page {item.resume_page}"
                st.caption(f"{sources} · Target evidence: {item.job_evidence}")

if st.session_state["rag_error"]:
    st.warning(
        "Semantic evidence retrieval was unavailable, so CareerFit kept the "
        f"Part 3 lexical responsibility score. {st.session_state['rag_error']}"
    )

if st.session_state["rag_result"]:
    rag = st.session_state["rag_result"]
    st.divider()
    st.header("Semantic evidence index")
    with st.container(horizontal=True):
        st.metric("Stored chunks", rag.chunk_count, border=True)
        st.metric("Vector dimensions", rag.embedding_dimension, border=True)
        st.metric(
            "Semantic requirements",
            len(rag.semantic_responsibility_matches),
            border=True,
        )
    st.caption(
        f"Analysis `{rag.analysis_id[:8]}…` uses `{rag.embedding_model}`. "
        "Every retrieval is filtered by the complete random analysis ID and "
        "résumé document type."
    )
    if st.button(
        "Delete analysis data",
        icon=":material/delete:",
        type="secondary",
    ):
        try:
            deleted = get_vector_store(
                rag.embedding_model, rag.embedding_dimension
            ).delete_analysis(rag.analysis_id)
        except VectorStoreError as exc:
            st.error(str(exc))
        else:
            clear_analysis_outputs()
            st.toast(f"Deleted {deleted} stored chunks.", icon=":material/delete:")
            st.rerun()
