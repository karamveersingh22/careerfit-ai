# CareerFit AI - Project Context and Engineering Handoff

Last updated: 2026-08-02

## 1. Project identity

- Project name: CareerFit AI
- Purpose: Compare one résumé PDF with one target job-description PDF and produce evidence-grounded fit analysis.
- Production application: https://careerfitai.streamlit.app/
- GitHub repository: https://github.com/karamveersingh22/careerfit-ai
- Local project path used during development: `E:\coding\gen ai\careerfit`
- Main branch: `main`
- Streamlit entry point: `app.py`
- Current implemented milestone: Parts 1 through 6

CareerFit is a structured document-comparison application. It does not claim to reproduce a proprietary applicant tracking system. Gemini extracts structured facts and writes grounded guidance; deterministic Python rules calculate every score.

## 2. Current deployment status

The Streamlit application is deployed and publicly reachable at:

```text
https://careerfitai.streamlit.app/
```

Verified live on 2026-08-02:

- The application loads successfully.
- The upload form is visible.
- The sidebar shows Parts 1-6.
- The Part 6 benchmark is available without using Gemini.
- The deployed app currently says `Gemini API key is not configured`.

To enable live PDF analysis, add these values in Streamlit Community Cloud under **App settings -> Secrets**:

```toml
GEMINI_API_KEY = "your-real-private-key"
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
GEMINI_EMBEDDING_DIMENSION = 768
```

Never put the real key in GitHub, this file, `app.py`, screenshots, logs, or chat messages.

## 3. Product inputs and outputs

### Inputs

- One text-based résumé PDF.
- One text-based job-description PDF.
- Explicit user consent before extracted text is sent to Gemini.

### Outputs

- Validated `ResumeProfile` and `JobProfile` JSON.
- Overall CareerFit match score.
- Separate content-readiness score.
- Component scores and published weights.
- Exact, equivalent, related, and missing skill matches.
- Page-linked responsibility evidence.
- Strengths and evidence gaps.
- Truthful résumé rewrite suggestions.
- Job-specific learning recommendations.
- Technical, résumé-specific, and gap-focused interview questions.
- A one-click action to delete the current analysis from Chroma.

## 4. High-level architecture

```text
Résumé PDF                         Job-description PDF
    |                                      |
    +---------- validation/extraction -----+
                       |
          page-aware ExtractedDocument objects
                       |
          +------------+-------------+
          |                          |
  Gemini structured extraction   section-aware chunking
          |                          |
 ResumeProfile + JobProfile      Gemini embeddings
          |                          |
          |                     local ChromaDB
          |                          |
          +------ comparison + retrieval
                       |
             deterministic scoring
                       |
             grounded Gemini guidance
                       |
                Streamlit dashboard
```

The architecture intentionally separates document handling, AI extraction, matching, scoring, retrieval, guidance, presentation, and evaluation. This allows each layer to be tested independently.

## 5. End-to-end runtime workflow

When the user clicks **Analyze documents**:

1. Streamlit checks that both files and the provider-consent checkbox are present.
2. `document_engine.extract_pdf()` validates and extracts both PDFs in memory.
3. `GeminiProfileExtractor.extract_pair()` sends page-marked text to Gemini twice: once using the résumé schema and once using the JD schema.
4. Pydantic validates both structured responses and source-page references.
5. `score_profiles()` calculates the first deterministic score using lexical responsibility evidence.
6. `chunk_document()` creates page-safe and section-aware chunks for both documents.
7. `GeminiEmbeddingClient` embeds all chunks.
8. `ChromaVectorStore` stores chunks, vectors, and metadata under a random analysis ID.
9. Every JD responsibility is embedded as a query and searched only against résumé chunks from the same analysis.
10. The stronger of lexical and semantic responsibility evidence is retained.
11. `score_profiles()` recalculates the deterministic score using the improved responsibility evidence.
12. `GeminiGuidanceGenerator` receives the final profiles, final scores, and page-marked documents.
13. Gemini produces strengths, gaps, rewrites, learning actions, and interview questions using a Pydantic response schema.
14. Every quoted citation is checked against the exact cited PDF page.
15. Streamlit saves results in per-session state and renders the dashboard.

If RAG fails, the application keeps the lexical responsibility result. If guidance fails, scores and evidence remain visible. Provider failures are isolated so a later layer does not destroy earlier successful results.

## 6. Project structure

```text
careerfit-ai/
|-- app.py                         Streamlit entry point and UI orchestration
|-- requirements.txt              Runtime and test dependencies
|-- README.md                      Public project overview
|-- project_context.md             This engineering handoff
|-- .gitignore                     Privacy and generated-file exclusions
|-- .streamlit/
|   |-- secrets.toml.example       Safe configuration template
|   `-- secrets.toml               Local private secrets; never committed
|-- careerfit/
|   |-- document_engine.py         PDF validation, cleanup, page extraction
|   |-- models.py                  Pydantic contracts for every application layer
|   |-- prompts.py                 Structured profile-extraction prompts
|   |-- profile_extraction.py      Gemini structured extraction adapter
|   |-- skill_matching.py          Deterministic skill normalization and credits
|   |-- scoring.py                 Published scoring and readiness rules
|   |-- chunking.py                Page-safe, section-aware chunks
|   |-- embeddings.py              Gemini embedding adapter
|   |-- vector_store.py            Chroma persistence, filtering, deletion
|   |-- rag.py                     Semantic responsibility retrieval and merge
|   |-- guidance.py                Grounded Part 5 guidance and citation validation
|   |-- presentation.py            Domain-model to dashboard-row conversion
|   `-- evaluation.py              Part 6 synthetic benchmark
`-- tests/                         Unit, regression, integration, and UI smoke tests
```

## 7. Module responsibilities and important logic

### `app.py`

- Configures the wide Streamlit page.
- Reads settings from environment variables first, then `st.secrets`.
- Keeps document and analysis results in `st.session_state`.
- Uses `st.cache_resource` for Gemini clients and Chroma stores.
- Uses `st.cache_data` for the deterministic Part 6 benchmark.
- Processes uploaded bytes without saving the original PDFs.
- Coordinates extraction, scoring, RAG, guidance, rendering, and deletion.
- Shows safe provider errors instead of raw exceptions or secrets.

Important session-state keys:

```text
resume_document
jd_document
profile_result
score_result
rag_result
rag_error
guidance_result
guidance_error
```

### `careerfit/document_engine.py`

Uses PyMuPDF (`fitz`) and enforces:

- `.pdf` filename extension.
- `%PDF-` file signature.
- Non-empty upload.
- Maximum size of 10 MB.
- Maximum 50 pages.
- No password-protected PDFs.
- At least 40 readable characters across the document.

Text is extracted page by page and cleaned conservatively. Page numbers are never discarded. Image-only PDFs are rejected because OCR is not implemented.

### `careerfit/models.py`

Contains frozen Pydantic models used as contracts between layers. Important groups are:

- Document models: `ExtractedPage`, `ExtractedDocument`.
- Evidence models: `EvidenceItem`, `SkillEvidence`.
- Profile models: `ResumeProfile`, `JobProfile`, `ProfileExtractionResult`.
- Scoring models: `SkillMatch`, `ResponsibilityMatch`, `ReadinessCheck`, `ScoringResult`.
- RAG models: `DocumentChunk`, `RetrievalHit`, `RagAnalysisResult`.
- Guidance models: `GuidanceItem`, `ResumeRewrite`, `LearningRecommendation`, `InterviewQuestion`, `CareerGuidanceResult`.

Unknown scalar values are represented as `None`; absent collections use empty lists. Evidence records retain page numbers.

### `careerfit/prompts.py` and `profile_extraction.py`

Gemini receives document text with visible page markers. Prompt rules require:

- Evidence only from the supplied document.
- No guessing or common-industry assumptions.
- Document instructions treated as data, not executable instructions.
- Required and preferred JD skills kept separate.
- Atomic skill extraction: one technology or concept per item.
- Source pages attached to extracted evidence.

Gemini uses schema-constrained JSON with `ResumeProfile` or `JobProfile` as the response schema. Invalid structured output can be retried, but provider, key, quota, and network failures are not repeatedly retried. Page references outside the document are rejected.

Default generation model:

```text
gemini-3.5-flash-lite
```

### `careerfit/skill_matching.py`

Skill matching is deterministic. It does not ask Gemini to decide match credit.

Normalization:

- Case folding.
- Punctuation and whitespace normalization.
- Known aliases such as JS/JavaScript, GCP/Google Cloud Platform, SQL/Structured Query Language, NLP/Natural Language Processing, and REST variants.
- Detection of an explicitly named résumé skill inside an accidentally long JD skill phrase.
- Word boundaries prevent partial-word false matches.

Credit rules:

| Match type | Credit |
|---|---:|
| Exact normalized skill | 1.0 |
| Known equivalent or named alias | 0.9 |
| Deliberately related technology | 0.5 |
| Missing evidence | 0.0 |

Related groups are deliberately narrow, such as Flask/Django/FastAPI, React/Angular/Vue, MySQL/PostgreSQL/SQLite, AWS/GCP/Azure, and PyTorch/TensorFlow/Keras. A related tool is never treated as the exact required tool.

Repeated requirements are deduplicated by canonical skill name before scoring.

### `careerfit/scoring.py`

Gemini never creates the CareerFit score. Python calculates it from these fixed weights:

| Component | Weight |
|---|---:|
| Required skills | 40% |
| Experience | 20% |
| Responsibilities | 15% |
| Preferred skills | 10% |
| Education | 10% |
| Résumé quality | 5% |

Formula:

```text
overall = sum(component_score * component_weight)
```

Experience logic:

```text
min(candidate_years / required_years, 1.0) * 100
```

If the JD does not state an experience requirement, the experience component is neutral at 100. If the requirement exists but candidate experience is unknown, it is 0.

Responsibility lexical matching uses meaningful token overlap between each JD responsibility and résumé work/project evidence. Part 4 can replace it with stronger semantic evidence.

Education matching recognizes degree levels such as diploma, bachelor, master, and doctorate, then supplements level comparison with lexical overlap.

Content-readiness is a separate score with these weights:

| Readiness check | Weight |
|---|---:|
| Contact details | 15% |
| Professional summary | 10% |
| Skills section | 15% |
| Experience or projects | 20% |
| Education | 10% |
| Clear headings | 10% |
| Action verbs | 5% |
| Quantified achievements | 10% |
| Relevant keywords | 5% |

The readiness score measures content completeness and clarity, not proprietary ATS parsing behaviour.

### `careerfit/chunking.py`

- Detects common résumé/JD headings and normalizes them into section names.
- Never creates a chunk that crosses a PDF page boundary.
- Default maximum chunk length is 700 characters.
- Splits long text by sentence and then by word when necessary.
- Gives every chunk a stable ID containing analysis ID, document type, page, and chunk number.

### `careerfit/embeddings.py`

Default configuration:

```text
Model: gemini-embedding-2
Dimension: 768
```

Document embeddings and query embeddings use asymmetric prefixes. Returned vectors are checked for count and dimension, then normalized to unit length before storage/search.

### `careerfit/vector_store.py`

Uses a local persistent Chroma client at `data/chroma`. Collection names include the normalized embedding model and dimension so incompatible vector spaces cannot be mixed.

Every record stores:

```text
analysis_id
document_type
section
page
chunk text
embedding
```

Every search requires both a random `analysis_id` and a `document_type` filter. Responsibility retrieval searches only résumé chunks belonging to the current analysis. Chroma uses cosine distance; the application converts it to a bounded similarity value with `1 - distance`.

`delete_analysis()` removes only records belonging to the selected analysis ID.

On Streamlit Community Cloud, local disk is temporary. Chroma data may disappear when the app restarts or hibernates. This is acceptable for the current session-oriented prototype because an index is rebuilt for every new analysis.

### `careerfit/rag.py`

- Creates a random UUID for every analysis.
- Chunks and embeds both documents.
- Stores all chunks with metadata.
- Embeds each major JD responsibility as a query.
- Retrieves the top three résumé chunks within the filtered analysis.
- Accepts semantic evidence only at similarity `0.45` or higher.
- Compares semantic and lexical evidence and retains the stronger score.
- Deletes partial analysis data when indexing or retrieval fails.

### `careerfit/guidance.py`

Part 5 asks Gemini for:

- Two to four non-repetitive strengths.
- Up to five gaps.
- Up to three truthful résumé rewrites.
- Up to five job-specific learning actions.
- Up to six interview questions split across technical, résumé, and gap categories.

The final deterministic score is supplied as read-only context. The prompt explicitly prohibits recalculation and invented achievements, numbers, tools, duties, qualifications, or experience.

Every evidence field must be a short verbatim excerpt with its page. After Gemini responds, the application normalizes whitespace and verifies that each quote actually exists on the cited page. Invalid citations can trigger a validation retry. Gaps are described as missing document evidence, not proof that the person lacks real ability.

### `careerfit/presentation.py`

Converts domain models to dictionaries for Streamlit tables. Keeping this separate prevents UI code from reading fields from the wrong model type. This module was added after a regression where responsibility-only fields were accidentally read from `SkillMatch` records.

### `careerfit/evaluation.py`

Part 6 is an offline, deterministic benchmark. It uses no Gemini quota.

- 24 synthetic résumé/JD pairs.
- Eight job families.
- Eight strong, eight medium, and eight weak cases.
- Score bands: strong >= 75, medium 45-74.9, weak < 45.
- Uses the production `score_profiles()` function unchanged.
- Checks band accuracy, balanced coverage, reproducibility, and score separation.

Current expected benchmark result:

```text
24/24 cases in the expected band
100.0% band accuracy
Strong average: approximately 99.2
Medium average: approximately 61.1
Weak average: approximately 0.5
```

This benchmark proves internal consistency on controlled fixtures. It does not prove equivalence to a real employer's ATS or accuracy on every real-world résumé.

## 8. User-interface sections

The application currently renders:

1. Sidebar configuration and privacy notices.
2. Part 6 quality benchmark.
3. Two PDF upload controls and provider consent.
4. Page/character extraction summaries and optional text previews.
5. Validated résumé and job profiles.
6. Transparent score summary cards.
7. Formula, skill matches, readiness, and responsibility-evidence tabs.
8. Grounded strengths, gaps, rewrites, learning, and interview tabs.
9. Semantic evidence-index metadata.
10. Delete-analysis action.

## 9. Configuration and secrets

Settings are read from environment variables first and Streamlit secrets second.

| Setting | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | None | Required private Gemini credential |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Structured extraction and guidance |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2` | Chunk and query embeddings |
| `GEMINI_EMBEDDING_DIMENSION` | `768` | Vector dimension and collection identity |

Local secrets file:

```text
.streamlit/secrets.toml
```

Safe template:

```text
.streamlit/secrets.toml.example
```

Changing the embedding model or dimension requires a new index. Collection names prevent vectors from different model spaces from being mixed.

## 10. Local development setup

Requirements:

- Python 3.12 recommended.
- Git.
- Internet access for Gemini generation and embeddings.
- A Gemini API key from Google AI Studio.

From PowerShell:

```powershell
cd "E:\coding\gen ai\careerfit"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Edit `.streamlit/secrets.toml` and replace only the key placeholder.

Start the application:

```powershell
python -m streamlit run app.py
```

The default local URL is normally:

```text
http://localhost:8501
```

## 11. Testing commands

Run all tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Current baseline:

```text
63 tests passed
```

Run only the Part 6 benchmark tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_evaluation.py -v
```

Run the benchmark as a console report:

```powershell
.\.venv\Scripts\python.exe -m careerfit.evaluation
```

Test categories cover PDF validation, models, structured extraction, skill matching, scoring, chunking, embeddings, Chroma filtering/deletion, RAG merging, presentation rows, grounded guidance, evaluation, and Streamlit startup.

Tests use fake provider clients or synthetic data unless an explicit manual live-provider test is performed. Normal automated tests do not spend Gemini quota.

## 12. Git workflow

Normal change workflow:

```powershell
git status
git add .
git commit -m "Describe the change"
git push
```

Before committing, verify that these never appear in staged changes:

```text
.streamlit/secrets.toml
.venv/
data/chroma/
personal résumé PDFs
private job-description PDFs
```

The repository's `main` branch tracks `origin/main`. Streamlit Community Cloud watches this branch and normally rebuilds or reruns the application after a push.

## 13. Streamlit Community Cloud deployment

Deployment coordinates:

```text
Repository: karamveersingh22/careerfit-ai
Branch: main
Entrypoint: app.py
Python: 3.12
Domain: https://careerfitai.streamlit.app/
```

To update cloud secrets:

1. Open https://share.streamlit.io/.
2. Locate the CareerFit app.
3. Open the app's menu and select **Settings**.
4. Open **Secrets**.
5. Paste valid TOML settings without committing them to GitHub.
6. Save and reboot if the app does not automatically restart.

Free Community Cloud apps can hibernate after inactivity. Visiting the URL wakes the app. Local Chroma files and Streamlit session state are not durable across restarts.

## 14. Privacy and security boundaries

- Uploaded PDF bytes are processed in memory and are not deliberately saved by the application.
- Extracted text is sent to Gemini only after explicit consent.
- The local/deployed Chroma store contains text chunks and embeddings until deletion or environment restart.
- Random analysis IDs isolate vector queries between analyses.
- Full document text is not intentionally logged.
- Provider exceptions are converted into safe user-facing messages.
- Free-tier Gemini content may be used by Google to improve its products; the UI warns users.
- A public deployment allows visitors to consume the owner's Gemini quota even though they cannot see the key.
- Keep the app private or add access/rate controls before sharing it widely.
- Review Google's current retention and usage terms before processing real personal résumés.

## 15. Failure behaviour

- Invalid/unreadable PDF: processing stops with a safe PDF error.
- Invalid Gemini profile JSON: validation retry, then safe extraction error.
- Provider/key/model/quota/network error: no repeated provider retry; show a safe message.
- Embedding or Chroma failure: retain lexical responsibility evidence when possible.
- Partial RAG index: attempt cleanup by analysis ID.
- Guidance failure: retain profiles, deterministic scores, and evidence.
- Citation mismatch: reject guidance and retry validation.
- Cloud restart: session state and local vector data may be lost; the user reruns analysis.

## 16. Known limitations

- Text-based PDFs only; no OCR for scanned/image-only documents.
- One résumé and one JD per analysis.
- No authentication, user accounts, saved history, or multi-user database.
- Chroma is local and ephemeral on Streamlit Community Cloud.
- The public app can consume the project owner's Gemini quota.
- Skill aliases and related-technology groups are curated, not exhaustive.
- Experience compares stated total experience with stated minimum experience; relevant-experience duration is not independently calculated.
- Semantic responsibility similarity is evidence retrieval, not proof of ability.
- Guidance quality depends on provider availability and quota.
- Benchmark cases validate deterministic behaviour, not external ATS validity.
- The application does not generate a finished résumé document.

## 17. Safe extension points

Recommended next improvements, in practical order:

1. Configure and verify production Gemini secrets.
2. Add deployment access control or per-user rate limiting.
3. Add local OCR fallback for scanned PDFs.
4. Replace local Chroma with a managed vector store for durable multi-user deployment.
5. Add explicit analysis cleanup on session expiration.
6. Add a downloadable, privacy-reviewed analysis report.
7. Expand skill taxonomy and evaluate it against labelled real-world examples.
8. Add relevant-experience scoring by role and responsibility evidence.
9. Add observability using counts/timings without logging résumé text.
10. Add end-to-end deployment smoke tests.

When extending the system, preserve these invariants:

- Gemini does not create the numeric score.
- Every recommendation claim is grounded in uploaded evidence.
- Missing evidence is not described as missing real-world ability.
- Every vector search is filtered by analysis and source type.
- Secrets and personal documents never enter Git history.
- New scoring rules have published weights and regression tests.

## 18. Handoff checklist

Before considering a new environment operational:

- [ ] `GEMINI_API_KEY` is configured outside GitHub.
- [ ] Generation model is `gemini-3.5-flash-lite` or an explicitly tested replacement.
- [ ] Embedding model and dimension match stored collection configuration.
- [ ] `python -m pytest -q` passes.
- [ ] `python -m careerfit.evaluation` reports 24/24.
- [ ] The Streamlit sidebar shows Gemini configured.
- [ ] A synthetic PDF pair completes extraction, scoring, RAG, and guidance.
- [ ] Citation pages match the uploaded documents.
- [ ] Delete-analysis removes the current Chroma records.
- [ ] `.streamlit/secrets.toml` is ignored by Git.
- [ ] Public sharing and Gemini quota exposure are intentional.

## 19. Important terminology

- CareerFit match: deterministic weighted résumé/JD comparison score.
- Content readiness: deterministic résumé completeness and keyword-clarity score.
- Exact/equivalent/related/missing: published skill-credit categories.
- RAG: retrieval-augmented generation; here it primarily retrieves résumé evidence for JD requirements.
- Analysis ID: random UUID used to isolate one analysis in Chroma.
- Grounded guidance: Gemini-written advice whose factual excerpts are validated against cited pages.
- Missing evidence: the résumé does not explicitly show a requirement; it is not a claim about the person's true ability.

## 20. Source of truth

For runtime behaviour, source code and tests are the final authority. If this handoff conflicts with the code:

1. Inspect `app.py` for orchestration and visible behaviour.
2. Inspect the relevant module under `careerfit/` for business rules.
3. Inspect the corresponding test under `tests/` for the expected contract.
4. Update this file in the same commit as any architectural, model, scoring, deployment, or privacy change.
