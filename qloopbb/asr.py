from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: Optional[str]
    duration_seconds: float


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Optional[Path] = None,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Run `python -m pip install -r requirements.txt`."
            ) from exc

        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(download_root) if download_root else None,
        )

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        translate_to_english: bool = False,
        vad_filter: bool = True,
    ) -> TranscriptionResult:
        task = "translate" if translate_to_english else "transcribe"
        segments, info = self._model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            vad_filter=vad_filter,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return TranscriptionResult(
            text=text,
            language=getattr(info, "language", None),
            duration_seconds=float(getattr(info, "duration", 0.0) or 0.0),
        )
