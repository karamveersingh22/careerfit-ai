# CareerFit AI

Repository: https://github.com/karamveersingh22/careerfit-ai

CareerFit AI compares a résumé PDF with a target job-description PDF. The
project is being built in independently testable parts so that document
handling, structured extraction, scoring, retrieval, and presentation remain
separate.

## Current milestone: Part 6

Part 1 provides:

- a Streamlit upload interface for one résumé and one job description;
- PDF signature, extension, size, page-count, encryption, and readability checks;
- in-memory extraction with PyMuPDF;
- conservative text cleanup;
- preserved page numbers for future evidence citations; and
- automated tests using synthetic PDFs rather than personal documents.

OCR is deliberately deferred. Image-only, empty, damaged, and password-protected
documents receive clear errors.

Part 2 adds:

- explicit `ResumeProfile` and `JobProfile` Pydantic contracts;
- Gemini structured JSON output through the official Google GenAI SDK;
- separate prompts for résumés and job descriptions;
- null-for-unknown and no-guessing rules;
- page references on extracted evidence;
- rejection of page references outside the uploaded document;
- one retry only when Gemini returns invalid structured data; and
- a provider-data-transfer confirmation before an API request.

The default model is `gemini-3.5-flash-lite`. Set `GEMINI_MODEL` to change it without
editing application code.

Part 3 adds a deterministic scoring layer. Gemini supplies structured evidence,
but Python calculates every score using published rules:

- required skills: 40%;
- experience: 20%;
- responsibilities: 15%;
- preferred skills: 10%;
- education: 10%; and
- résumé quality: 5%.

Skill credit is exact `1.0`, known equivalent `0.9`, deliberately related `0.5`,
or missing `0.0`. Required skills are deduplicated after normalization. An absent
job requirement receives a neutral component score of 100 rather than penalizing
the candidate.

Content readiness is reported separately and checks contact details, summary,
skills, experience or projects, education, headings, action verbs, quantified
achievements, and relevant keywords. This is not a claim to reproduce a
proprietary ATS.

Responsibility matching begins with explainable word overlap and page-linked
résumé evidence. Part 4 supplements it with filtered semantic retrieval and keeps
whichever evidence method produces the stronger score.

Part 4 adds a local retrieval-augmented evidence layer:

- page-safe, section-aware chunks for both documents;
- Gemini `gemini-embedding-2` vectors at 768 dimensions;
- a persistent local Chroma collection configured for cosine distance;
- metadata containing random analysis ID, document type, section, and page;
- mandatory filtering by analysis ID and résumé source during evidence search;
- semantic responsibility retrieval with exact résumé and job pages;
- a 45% minimum cosine-similarity threshold before evidence is accepted;
- selection of the stronger lexical or semantic responsibility evidence; and
- one-click deletion of every Chroma record belonging to the analysis.

Part 5 adds a grounded career-guidance layer after scoring and retrieval:

- non-repetitive strengths supported by résumé and job evidence;
- gaps described as missing document evidence rather than missing real ability;
- truthful résumé rewrites that are forbidden from inventing achievements;
- job-specific learning actions;
- technical, résumé-specific, and gap-focused interview questions;
- schema-constrained Gemini output with exact page-linked excerpts;
- rejection and retry when a quoted excerpt is absent from its cited page; and
- failure isolation so guidance quota or validation errors never remove the
  deterministic scores and evidence already produced.

Gemini explains the completed analysis but cannot modify or supply its scores.
The dashboard separates strengths, gaps, rewrites, learning, and interviews so
each kind of recommendation has a clear purpose.

Part 6 adds a reproducible quality benchmark with 24 synthetic résumé/JD pairs:

- eight job families: backend, frontend, data science, cloud, mobile, security,
  data engineering, and full-stack development;
- one strong, medium, and weak candidate fixture for every job family;
- explicit score bands: strong at 75 or above, medium from 45 to 74.9, and weak
  below 45;
- production scoring code used unchanged for every benchmark case;
- automated checks for balanced coverage, score separation, band accuracy, and
  reproducibility; and
- a cached benchmark summary in the Streamlit sidebar that uses no Gemini quota.

Run the benchmark directly with `python -m careerfit.evaluation`.

Embedding model spaces are not interchangeable. If the embedding model or
dimension changes, documents must be embedded again; the collection name includes
both settings to prevent incompatible vectors from being mixed.

## Configure Gemini

1. Sign in to [Google AI Studio](https://aistudio.google.com/).
2. Open the API Keys page and create a new Gemini authorization key.
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
4. Replace the placeholder with the new key:

```toml
GEMINI_API_KEY = "your-private-key"
GEMINI_MODEL = "gemini-3.5-flash-lite"
```

5. Restart Streamlit after changing secrets.

Never paste the key into `app.py`, commit `secrets.toml`, or send the key through
chat. For deployments, set `GEMINI_API_KEY` in the hosting provider's secret or
environment-variable settings instead.

## Run locally

Create a virtual environment, install the requirements, and start Streamlit:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then upload a readable résumé PDF and job-description PDF, acknowledge the
provider data transfer, and start extraction in the browser.

## Run tests

```powershell
python -m pytest
```

## Privacy boundary

Uploaded bytes are processed in memory and are not written to disk by the
application. Full extracted document text is neither printed nor logged. During
Part 2, extracted text is sent to Gemini after the user confirms the provider
notice. Google states that free-tier content may be used to improve its products;
use synthetic documents during development and review current retention terms
before using real personal data. Local PDFs, API secrets, temporary files, and
the future Chroma database are excluded from version control.
