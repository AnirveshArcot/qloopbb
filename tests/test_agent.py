import unittest

from qloopbb.agent import SAFE_RECEPTIONIST_FALLBACK, build_reply
from qloopbb.embeddings import SearchResult, VectorDocument
from qloopbb.router import RouteDecision, RouteKind
from qloopbb.tools import ToolResult


class AgentTests(unittest.TestCase):
    def test_build_reply_keeps_fixed_fallback_without_route(self) -> None:
        self.assertEqual(build_reply("hello"), "that's cool")
        self.assertEqual(build_reply("मेरी appointment चाहिए"), "that's cool")

    def test_build_reply_asks_for_repeat_on_empty_route(self) -> None:
        reply = build_reply(
            "",
            route=RouteDecision(
                kind=RouteKind.EMPTY,
                confidence=1.0,
                reason="No usable transcript text.",
            ),
        )

        self.assertEqual(reply, "I didn't catch that. Could you say it again?")

    def test_build_reply_uses_safe_handoff_for_medical_advice(self) -> None:
        reply = build_reply(
            "what medicine should I take",
            route=RouteDecision(
                kind=RouteKind.MEDICAL_ADVICE,
                confidence=0.9,
                reason="clinical advice request",
                safe_handoff=True,
            ),
        )

        self.assertEqual(reply, SAFE_RECEPTIONIST_FALLBACK)

    def test_build_reply_uses_tool_result(self) -> None:
        reply = build_reply(
            "schedule an MRI",
            route=RouteDecision(
                kind=RouteKind.APPOINTMENT,
                confidence=0.8,
                reason="Scheduling keyword matched.",
                tool_name="appointment_lookup",
            ),
            tool_result=ToolResult(
                name="appointment_lookup",
                summary="Likely scheduling destination is Radiology.",
                data={"department": "Radiology"},
            ),
        )

        self.assertEqual(
            reply,
            "I can help with that. Likely scheduling destination is Radiology.",
        )

    def test_build_reply_uses_retrieval_result(self) -> None:
        route = RouteDecision(
            kind=RouteKind.DEPARTMENT_LOOKUP,
            confidence=0.75,
            reason="Department or symptom routing keyword matched.",
            needs_retrieval=True,
        )
        result = SearchResult(
            document=VectorDocument(
                id="1",
                text="Orthopedics handles knee appointments.",
                metadata={},
            ),
            score=0.9,
        )

        reply = build_reply("my knee hurts", route=route, retrieval_results=[result])

        self.assertEqual(
            reply,
            "I found this routing note: Orthopedics handles knee appointments.",
        )


if __name__ == "__main__":
    unittest.main()
