import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Optional, Protocol

from qloopbb.llm import (
    DEFAULT_GEMINI_BASE_URL,
    DEFAULT_GEMINI_MODEL,
    get_gemini_api_key,
    normalize_gemini_model,
    parse_gemini_text_response,
)
from qloopbb.net import build_https_context
from qloopbb.tts import normalize_language


class OutputLocalizationError(RuntimeError):
    pass


class OutputLocalizer(Protocol):
    def localize(self, text: str, target_language: str) -> str:
        pass


class NoopOutputLocalizer:
    def localize(self, text: str, target_language: str) -> str:
        return text


class GeminiOutputLocalizer:
    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_GEMINI_BASE_URL,
        temperature: float = 0.0,
        max_output_tokens: int = 120,
        timeout_seconds: float = 8.0,
    ) -> None:
        self.api_key = api_key or get_gemini_api_key()
        if not self.api_key:
            raise RuntimeError("Set GEMINI_API_KEY before using Gemini output localization.")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than 0")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

        self.model = normalize_gemini_model(model)
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds

    def localize(self, text: str, target_language: str) -> str:
        stripped_text = text.strip()
        if not stripped_text or is_english_language(target_language):
            return text

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": build_localization_prompt(
                                text=stripped_text,
                                target_language=target_language,
                            )
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        response = self._post_generate_content(payload)
        localized = parse_gemini_text_response(response)
        return localized or text

    def _post_generate_content(self, payload: Dict[str, object]) -> Dict[str, object]:
        model_path = urllib.parse.quote(self.model, safe="/")
        query = urllib.parse.urlencode({"key": self.api_key})
        url = f"{self.base_url}/v1beta/{model_path}:generateContent?{query}"
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
                context=build_https_context(),
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise OutputLocalizationError(
                f"Gemini localization failed with HTTP {exc.code}: {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OutputLocalizationError(f"Gemini localization failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise OutputLocalizationError("Gemini localization timed out.") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OutputLocalizationError("Gemini localization returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise OutputLocalizationError(
                "Gemini localization returned an unexpected response shape."
            )
        return parsed


def is_english_language(language: str) -> bool:
    normalized = normalize_language(language)
    root = normalized.split("-")[0].split("_")[0]
    return root == "en"


def build_localization_prompt(text: str, target_language: str) -> str:
    language_name = language_display_name(target_language)
    return "\n".join(
        [
            "Translate the receptionist reply into the target language.",
            "Preserve the exact meaning, urgency, and safety boundaries.",
            "Do not add medical advice, diagnosis, treatment guidance, or new facts.",
            "Keep names, department names, building locations, numbers, and codes unchanged unless translation is natural.",
            "Return only the translated spoken reply with no explanation.",
            f"Target language: {language_name} ({target_language})",
            f"English reply: {text}",
        ]
    )


def language_display_name(language: str) -> str:
    normalized = normalize_language(language)
    root = normalized.split("-")[0].split("_")[0]
    names = {
        "as": "Assamese",
        "bn": "Bengali",
        "en": "English",
        "gu": "Gujarati",
        "hi": "Hindi",
        "kn": "Kannada",
        "ml": "Malayalam",
        "mr": "Marathi",
        "ne": "Nepali",
        "or": "Odia",
        "pa": "Punjabi",
        "ta": "Tamil",
        "te": "Telugu",
        "ur": "Urdu",
    }
    return names.get(root, language)
