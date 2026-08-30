from dataclasses import dataclass
from typing import Optional

from qloopbb.asr import TranscriptionResult


@dataclass(frozen=True)
class TurnState:
    source_language: Optional[str]
    working_language: str
    response_language: str
    working_text: str


def build_turn_state(
    transcription: TranscriptionResult,
    translated_to_english: bool,
    response_language_override: Optional[str] = None,
) -> TurnState:
    source_language = transcription.language
    response_language = response_language_override or source_language or "en"
    working_language = "en" if translated_to_english else source_language or "unknown"

    return TurnState(
        source_language=source_language,
        working_language=working_language,
        response_language=response_language,
        working_text=transcription.text,
    )
