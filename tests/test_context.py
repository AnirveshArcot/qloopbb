import unittest

from qloopbb.context import resolve_route_context
from qloopbb.embeddings import SearchResult, VectorDocument
from qloopbb.retrieval import normalize_query_text
from qloopbb.router import route_utterance
from qloopbb.tools import LocalHospitalTools


class FakeRetriever:
    def __init__(self) -> None:
        self.calls = []

    def search(self, query, top_k=3):
        self.calls.append((query, top_k))
        return [
            SearchResult(
                document=VectorDocument(
                    id="1",
                    text="Orthopedics handles knee appointments.",
                    metadata={},
                ),
                score=0.9,
            )
        ]


class ContextTests(unittest.TestCase):
    def test_appointment_uses_tool_without_retrieval(self) -> None:
        retriever = FakeRetriever()
        route = route_utterance("I need to schedule an MRI")

        resolution = resolve_route_context(
            route=route,
            transcript="I need to schedule an MRI",
            tools=LocalHospitalTools(),
            retriever=retriever,
            top_k=3,
        )

        self.assertIsNotNone(resolution.tool_result)
        self.assertEqual(resolution.tool_result.name, "appointment_lookup")
        self.assertIsNone(resolution.retrieval_results)
        self.assertEqual(retriever.calls, [])

    def test_department_lookup_uses_tool_and_retrieval(self) -> None:
        retriever = FakeRetriever()
        route = route_utterance("My knee hurts")

        resolution = resolve_route_context(
            route=route,
            transcript="My knee hurts",
            tools=LocalHospitalTools(),
            retriever=retriever,
            top_k=1,
        )

        self.assertIsNotNone(resolution.tool_result)
        self.assertEqual(resolution.tool_result.name, "department_lookup")
        self.assertIsNotNone(resolution.retrieval_results)
        self.assertEqual(retriever.calls, [("My knee hurts", 1)])

    def test_matching_prewarmed_results_are_reused(self) -> None:
        retriever = FakeRetriever()
        route = route_utterance("My knee hurts")
        prewarmed_results = [
            SearchResult(
                document=VectorDocument(
                    id="1",
                    text="Orthopedics handles knee appointments.",
                    metadata={},
                ),
                score=0.9,
            )
        ]

        resolution = resolve_route_context(
            route=route,
            transcript="My knee hurts",
            tools=LocalHospitalTools(),
            retriever=retriever,
            top_k=1,
            prewarmed_results=prewarmed_results,
            prewarmed_query_key=normalize_query_text("My knee hurts"),
        )

        self.assertTrue(resolution.reused_prewarmed_results)
        self.assertEqual(resolution.retrieval_results, prewarmed_results)
        self.assertEqual(retriever.calls, [])


if __name__ == "__main__":
    unittest.main()
