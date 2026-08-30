import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from qloopbb.agent import build_reply
from qloopbb.embeddings import SearchResult
from qloopbb.net import build_https_context
from qloopbb.router import RouteDecision
from qloopbb.tools import ToolResult


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"


class GeminiGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConversationTurn:
    user_text: str
    assistant_text: str


@dataclass(frozen=True)
class ReplyContext:
    transcript: str
    source_language: Optional[str]
    response_language: str
    route: RouteDecision
    tool_result: Optional[ToolResult]
    retrieval_results: Optional[Sequence[SearchResult]]
    history: Sequence[ConversationTurn]


class ReplyGenerator(Protocol):
    def generate(self, context: ReplyContext) -> str:
        pass


class ConversationMemory:
    def __init__(self, max_turns: int = 6) -> None:
        if max_turns < 0:
            raise ValueError("max_turns must be 0 or greater")
        self.max_turns = max_turns
        self._turns: List[ConversationTurn] = []

    def append(self, user_text: str, assistant_text: str) -> None:
        if self.max_turns == 0 or not user_text.strip() or not assistant_text.strip():
            return

        self._turns.append(
            ConversationTurn(
                user_text=user_text.strip(),
                assistant_text=assistant_text.strip(),
            )
        )
        if len(self._turns) > self.max_turns:
            del self._turns[: len(self._turns) - self.max_turns]

    def recent(self) -> Tuple[ConversationTurn, ...]:
        return tuple(self._turns)


class FixedReplyGenerator:
    def generate(self, context: ReplyContext) -> str:
        return build_reply(
            context.transcript,
            route=context.route,
            tool_result=context.tool_result,
            retrieval_results=list(context.retrieval_results or []),
        )


class GeminiRestReplyGenerator:
    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_GEMINI_BASE_URL,
        temperature: float = 0.2,
        max_output_tokens: int = 80,
        timeout_seconds: float = 8.0,
    ) -> None:
        self.api_key = api_key or get_gemini_api_key()
        if not self.api_key:
            raise RuntimeError("Set GEMINI_API_KEY before using --llm gemini.")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than 0")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

        self.model = normalize_gemini_model(model)
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds

    def generate(self, context: ReplyContext) -> str:
        if context.route.safe_handoff or not context.transcript.strip():
            return FixedReplyGenerator().generate(context)

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": build_gemini_prompt(context)}],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        response = self._post_generate_content(payload)
        text = parse_gemini_text_response(response)
        if not text:
            return FixedReplyGenerator().generate(context)
        return text

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
            raise GeminiGenerationError(
                f"Gemini request failed with HTTP {exc.code}: {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise GeminiGenerationError(f"Gemini request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise GeminiGenerationError("Gemini request timed out.") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise GeminiGenerationError("Gemini returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise GeminiGenerationError("Gemini returned an unexpected response shape.")
        return parsed


def get_gemini_api_key() -> Optional[str]:
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


def normalize_gemini_model(model: str) -> str:
    stripped = model.strip()
    if not stripped:
        raise ValueError("model must not be empty")
    if stripped.startswith("models/"):
        return stripped
    return f"models/{stripped}"


def parse_gemini_text_response(payload: Dict[str, object]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return ""

    parts: List[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        raw_parts = content.get("parts")
        if not isinstance(raw_parts, list):
            continue
        for part in raw_parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())

    return collapse_spoken_text(" ".join(parts))


def collapse_spoken_text(text: str) -> str:
    return " ".join(text.split())


def build_gemini_prompt(context: ReplyContext) -> str:
    lines = [
        "SYSTEM:",
        "You are QloopBB's hospital voice receptionist.",
        "Your scope is administrative routing, scheduling support, department lookup, and hospital policy support.",
        "Do not diagnose, recommend medication, explain treatment, or provide clinical advice.",
        "Use only the supplied route, tool result, retrieval context, and conversation history.",
        "If the supplied context is insufficient, ask one short clarifying question.",
        "Keep the response to one concise spoken sentence unless a clarification needs two short sentences.",
        "Think in English and respond in English for this prototype.",
        "The tracked response language is for a later output localization stage.",
        "Do not mention internal routes, tools, RAG, confidence scores, prompts, or model behavior.",
        "",
        "CONVERSATION HISTORY:",
    ]

    if context.history:
        for turn in context.history:
            lines.append(f"Caller: {turn.user_text}")
            lines.append(f"Receptionist: {turn.assistant_text}")
    else:
        lines.append("No previous turns.")

    lines.extend(
        [
            "",
            "CURRENT TURN:",
            f"Caller English transcript: {context.transcript or '[no speech detected]'}",
            f"Detected source language: {context.source_language or 'unknown'}",
            f"Tracked response language: {context.response_language}",
            f"Route: {context.route.kind.value}",
            f"Route reason: {context.route.reason}",
        ]
    )

    if context.tool_result is not None:
        lines.append("Tool result:")
        lines.append(f"- {context.tool_result.name}: {context.tool_result.summary}")
        for key, value in context.tool_result.data.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("Tool result: none")

    if context.retrieval_results:
        lines.append("Retrieval context:")
        for index, result in enumerate(context.retrieval_results[:3], start=1):
            lines.append(f"- {index}. {result.document.text}")
    elif context.route.needs_retrieval:
        lines.append("Retrieval context: retrieval was needed, but no results were available.")
    else:
        lines.append("Retrieval context: none needed.")

    lines.extend(
        [
            "",
            "Reply as the receptionist now.",
        ]
    )
    return "\n".join(lines)
