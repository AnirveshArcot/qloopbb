import unittest

from qloopbb.asr import TranscriptionResult
from qloopbb.turns import build_turn_state


class TurnStateTests(unittest.TestCase):
    def test_translation_keeps_source_language_for_response(self) -> None:
        transcription = TranscriptionResult(
            text="I need a knee doctor",
            language="hi",
            duration_seconds=2.0,
        )

        turn_state = build_turn_state(transcription, translated_to_english=True)

        self.assertEqual(turn_state.source_language, "hi")
        self.assertEqual(turn_state.working_language, "en")
        self.assertEqual(turn_state.response_language, "hi")
        self.assertEqual(turn_state.working_text, "I need a knee doctor")

    def test_response_language_override_wins(self) -> None:
        transcription = TranscriptionResult(
            text="I need an MRI",
            language="ta",
            duration_seconds=2.0,
        )

        turn_state = build_turn_state(
            transcription,
            translated_to_english=True,
            response_language_override="en",
        )

        self.assertEqual(turn_state.response_language, "en")


if __name__ == "__main__":
    unittest.main()
