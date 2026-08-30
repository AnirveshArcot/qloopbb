import unittest

from qloopbb.embeddings import SearchResult, VectorDocument
from qloopbb.llm import (
    ConversationMemory,
    ConversationTurn,
    GeminiRestReplyGenerator,
    ReplyContext,
    build_gemini_prompt,
    normalize_gemini_model,
    parse_gemini_text_response,
)
from qloopbb.router import RouteDecision, RouteKind, route_utterance
from qloopbb.tools import ToolResult


class LlmTests(unittest.TestCase):
    def test_conversation_memory_keeps_recent_turns(self) -> None:
        memory = ConversationMemory(max_turns=2)

        memory.append("first", "one")
        memory.append("second", "two")
        memory.append("third", "three")

        self.assertEqual(
            memory.recent(),
            (
                ConversationTurn(user_text="second", assistant_text="two"),
                ConversationTurn(user_text="third", assistant_text="three"),
            ),
        )

    def test_parse_gemini_text_response_collapses_parts(self) -> None:
        parsed = parse_gemini_text_response(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Let me check\n"},
                                {"text": "that for you."},
                            ]
                        }
                    }
                ]
            }
        )

        self.assertEqual(parsed, "Let me check that for you.")

    def test_normalize_gemini_model_adds_models_prefix(self) -> None:
        self.assertEqual(
            normalize_gemini_model("gemini-2.5-flash"),
            "models/gemini-2.5-flash",
        )
        self.assertEqual(
            normalize_gemini_model("models/gemini-2.5-flash"),
            "models/gemini-2.5-flash",
        )

    def test_prompt_includes_history_tools_and_retrieval(self) -> None:
        context = ReplyContext(
            transcript="I need an MRI appointment",
            source_language="en",
            response_language="en",
            route=route_utterance("I need an MRI appointment"),
            tool_result=ToolResult(
                name="appointment_lookup",
                summary="Likely scheduling destination is Radiology.",
                data={"department": "Radiology"},
            ),
            retrieval_results=[
                SearchResult(
                    document=VectorDocument(
                        id="1",
                        text="Radiology schedules MRI appointments.",
                        metadata={},
                    ),
                    score=0.8,
                )
            ],
            history=[
                ConversationTurn(
                    user_text="hello",
                    assistant_text="that's cool",
                )
            ],
        )

        prompt = build_gemini_prompt(context)

        self.assertIn("Caller: hello", prompt)
        self.assertIn("appointment_lookup", prompt)
        self.assertIn("Radiology schedules MRI appointments.", prompt)
        self.assertIn("Do not diagnose", prompt)

    def test_gemini_skips_network_for_safe_handoff(self) -> None:
        class NoNetworkGemini(GeminiRestReplyGenerator):
            def _post_generate_content(self, payload):
                raise AssertionError("network should not be called")

        route = RouteDecision(
            kind=RouteKind.MEDICAL_ADVICE,
            confidence=0.9,
            reason="clinical advice request",
            safe_handoff=True,
        )
        generator = NoNetworkGemini(api_key="test")

        reply = generator.generate(
            ReplyContext(
                transcript="what medicine should I take",
                source_language="en",
                response_language="en",
                route=route,
                tool_result=None,
                retrieval_results=None,
                history=(),
            )
        )

        self.assertEqual(
            reply,
            "I am a receptionist, let me connect you with a triage nurse.",
        )


if __name__ == "__main__":
    unittest.main()
