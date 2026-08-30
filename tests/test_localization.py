import unittest

from qloopbb.localization import (
    GeminiOutputLocalizer,
    NoopOutputLocalizer,
    build_localization_prompt,
    is_english_language,
    language_display_name,
)


class LocalizationTests(unittest.TestCase):
    def test_noop_localizer_returns_original_text(self) -> None:
        text = "I can help with that."

        self.assertEqual(NoopOutputLocalizer().localize(text, "hi"), text)

    def test_english_language_detection_handles_region_codes(self) -> None:
        self.assertTrue(is_english_language("en"))
        self.assertTrue(is_english_language("en-US"))
        self.assertFalse(is_english_language("hi"))

    def test_language_display_name_for_indian_language(self) -> None:
        self.assertEqual(language_display_name("hi-IN"), "Hindi")
        self.assertEqual(language_display_name("ta"), "Tamil")

    def test_prompt_is_translation_only(self) -> None:
        prompt = build_localization_prompt(
            text="I can help with that.",
            target_language="hi",
        )

        self.assertIn("Translate the receptionist reply", prompt)
        self.assertIn("Do not add medical advice", prompt)
        self.assertIn("Hindi (hi)", prompt)
        self.assertIn("I can help with that.", prompt)

    def test_gemini_localizer_skips_english(self) -> None:
        class NoNetworkLocalizer(GeminiOutputLocalizer):
            def _post_generate_content(self, payload):
                raise AssertionError("network should not be called")

        localizer = NoNetworkLocalizer(api_key="test")

        self.assertEqual(
            localizer.localize("I can help with that.", "en-US"),
            "I can help with that.",
        )

    def test_gemini_localizer_returns_translated_text(self) -> None:
        class FakeGeminiLocalizer(GeminiOutputLocalizer):
            def _post_generate_content(self, payload):
                prompt = payload["contents"][0]["parts"][0]["text"]
                self.last_prompt = prompt
                return {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "मैं इसमें मदद कर सकता हूं।"},
                                ]
                            }
                        }
                    ]
                }

        localizer = FakeGeminiLocalizer(api_key="test")

        translated = localizer.localize("I can help with that.", "hi")

        self.assertEqual(translated, "मैं इसमें मदद कर सकता हूं।")
        self.assertIn("Target language: Hindi (hi)", localizer.last_prompt)


if __name__ == "__main__":
    unittest.main()
