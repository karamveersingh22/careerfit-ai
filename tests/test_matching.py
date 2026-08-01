import unittest

from careerfit.matching import ResumeJobMatcher, chunk_text


class FakeEmbeddingModel:
    def embed(self, texts):
        return [[float(len(text))] for text in texts]


class FakeExplanationModel:
    def explain(self, job_description, evidence):
        return f"Found {len(evidence)} evidence items"


class FakeCollection:
    def __init__(self):
        self.documents = []

    def add(self, ids, documents, embeddings):
        self.documents = list(documents)

    def query(self, query_embeddings, n_results, include):
        docs = self.documents[:n_results]
        distances = [float(i) for i in range(len(docs))]
        return {"documents": [docs], "distances": [distances]}


class MatchingTests(unittest.TestCase):
    def test_chunk_text_splits_long_input(self):
        text = "word " * 300
        chunks = chunk_text(text, chunk_size=80)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 80 for chunk in chunks))

    def test_match_returns_evidence_and_explanation(self):
        matcher = ResumeJobMatcher(FakeCollection(), FakeEmbeddingModel(), FakeExplanationModel())
        evidence, explanation = matcher.match(
            "Python SQL ML streamlit " * 220,
            "Looking for Python and SQL",
            top_k=3,
        )

        self.assertEqual(len(evidence), 3)
        self.assertEqual(explanation, "Found 3 evidence items")
        self.assertTrue(evidence[0].score >= evidence[1].score)


if __name__ == "__main__":
    unittest.main()
