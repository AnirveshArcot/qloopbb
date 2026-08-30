import unittest

import numpy as np

from qloopbb.embeddings import InMemoryVectorIndex, normalize_matrix, normalize_vector
from qloopbb.retrieval import normalize_query_text


class FakeEmbeddingModel:
    def embed_documents(self, texts):
        vectors = [self._embed(text) for text in texts]
        return normalize_matrix(vectors)

    def embed_query(self, text):
        return normalize_vector(self._embed(text))

    def _embed(self, text):
        lowered = text.lower()
        return np.array(
            [
                float("knee" in lowered or "orthopedic" in lowered),
                float("mri" in lowered or "radiology" in lowered),
                float("billing" in lowered or "insurance" in lowered),
            ],
            dtype=np.float32,
        )


class EmbeddingIndexTests(unittest.TestCase):
    def test_search_returns_most_similar_document(self) -> None:
        model = FakeEmbeddingModel()
        index = InMemoryVectorIndex.from_texts(
            texts=[
                "Orthopedics handles knee pain.",
                "Radiology schedules MRI scans.",
                "Billing answers insurance questions.",
            ],
            embedding_model=model,
        )

        results = index.search("I need an MRI appointment", model, top_k=1)

        self.assertEqual(results[0].document.text, "Radiology schedules MRI scans.")

    def test_rejects_empty_documents(self) -> None:
        model = FakeEmbeddingModel()

        with self.assertRaises(ValueError):
            InMemoryVectorIndex.from_texts([], model)

    def test_normalize_query_text_removes_case_and_punctuation(self) -> None:
        self.assertEqual(
            normalize_query_text(" I need an MRI, please! "),
            "i need an mri please",
        )


if __name__ == "__main__":
    unittest.main()
