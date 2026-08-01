# careerfit-ai

Evidence-grounded resume and job-description matching application built with Streamlit, Gemini and ChromaDB.

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your_api_key"
streamlit run app.py
```

If `GOOGLE_API_KEY` is missing, the app still returns evidence chunks and a rule-based summary.

## How it works

1. Resume text is chunked into smaller passages.
2. ChromaDB stores chunk embeddings generated with Gemini embeddings.
3. Job description embedding is queried against ChromaDB to retrieve top evidence chunks.
4. Gemini generates a concise explanation grounded only in retrieved evidence.

## Tests

```bash
python -m unittest discover -s tests
```
