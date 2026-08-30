import subprocess
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class AmbientMixConfig:
    audio_path: Optional[Path] = None
    ambient_volume: float = 0.04
    speech_volume: float = 1.0
    ducking: float = 0.35


class TextToSpeechEngine(Protocol):
    def speak_async(
        self,
        text: str,
        language: str = "en",
        is_thinking: bool = False,
    ) -> "TtsHandle":
        pass

    def speak(
        self,
        text: str,
        language: str = "en",
        is_thinking: bool = False,
    ) -> None:
        pass


class ProcessSlot:
    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def set(self, process: Optional[subprocess.Popen]) -> None:
        with self._lock:
            self._process = process

    def terminate(self) -> None:
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()


@dataclass
class TtsHandle:
    thread: Optional[threading.Thread] = None
    stop_event: Optional[threading.Event] = None
    process_slot: Optional[ProcessSlot] = None

    def wait(self) -> None:
        if self.thread is not None:
            self.thread.join()

    def stop(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.process_slot is not None:
            self.process_slot.terminate()


class NoopTtsEngine:
    def speak_async(
        self,
        text: str,
        language: str = "en",
        is_thinking: bool = False,
    ) -> TtsHandle:
        return TtsHandle()

    def speak(
        self,
        text: str,
        language: str = "en",
        is_thinking: bool = False,
    ) -> None:
        self.speak_async(text=text, language=language, is_thinking=is_thinking).wait()


class PiperCliTtsEngine:
    def __init__(
        self,
        model: str,
        binary: str = "piper",
        model_by_language: Optional[Dict[str, str]] = None,
        speaker: Optional[str] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        ambient: Optional[AmbientMixConfig] = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if not binary.strip():
            raise ValueError("binary must not be empty")

        self.model = model
        self.binary = binary
        self.model_by_language = {
            language.casefold(): voice_model
            for language, voice_model in (model_by_language or {}).items()
        }
        self.speaker = speaker
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.ambient = ambient or AmbientMixConfig()

    def speak_async(
        self,
        text: str,
        language: str = "en",
        is_thinking: bool = False,
    ) -> TtsHandle:
        stripped_text = text.strip()
        if not stripped_text:
            return TtsHandle()

        stop_event = threading.Event()
        process_slot = ProcessSlot()
        thread = threading.Thread(
            target=self._synthesize_and_play,
            args=(stripped_text, language, stop_event, process_slot),
            daemon=True,
        )
        thread.start()
        return TtsHandle(
            thread=thread,
            stop_event=stop_event,
            process_slot=process_slot,
        )

    def speak(
        self,
        text: str,
        language: str = "en",
        is_thinking: bool = False,
    ) -> None:
        self.speak_async(text=text, language=language, is_thinking=is_thinking).wait()

    def select_model(self, language: str) -> str:
        normalized = normalize_language(language)
        if normalized in self.model_by_language:
            return self.model_by_language[normalized]

        root_language = normalized.split("-")[0].split("_")[0]
        if root_language in self.model_by_language:
            return self.model_by_language[root_language]

        return self.model

    def _synthesize_and_play(
        self,
        text: str,
        language: str,
        stop_event: threading.Event,
        process_slot: ProcessSlot,
    ) -> None:
        if stop_event.is_set():
            return

        with tempfile.TemporaryDirectory(prefix="qloopbb-tts-") as temp_dir:
            output_path = Path(temp_dir) / "speech.wav"
            if not self._synthesize_to_wav(
                text=text,
                language=language,
                output_path=output_path,
                stop_event=stop_event,
                process_slot=process_slot,
            ):
                return
            if stop_event.is_set():
                return

            samples, sample_rate = read_wav_mono(output_path)
            mixed_samples = mix_ambient(samples, sample_rate, self.ambient)
            play_samples(mixed_samples, sample_rate, stop_event)

    def _synthesize_to_wav(
        self,
        text: str,
        language: str,
        output_path: Path,
        stop_event: threading.Event,
        process_slot: ProcessSlot,
    ) -> bool:
        command = self._build_command(
            model=self.select_model(language),
            output_path=output_path,
        )
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            print(f"TTS error: Piper binary not found: {self.binary}")
            return False

        process_slot.set(process)
        try:
            _, stderr = process.communicate(input=f"{text}\n".encode("utf-8"))
        finally:
            process_slot.set(None)

        if stop_event.is_set():
            return False
        if process.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            print(f"TTS error: Piper failed with code {process.returncode}: {message}")
            return False
        return output_path.exists()

    def _build_command(self, model: str, output_path: Path) -> List[str]:
        command = [
            self.binary,
            "--model",
            model,
            "--output_file",
            str(output_path),
        ]
        if self.speaker is not None:
            command.extend(["--speaker", self.speaker])
        if self.length_scale is not None:
            command.extend(["--length_scale", str(self.length_scale)])
        if self.noise_scale is not None:
            command.extend(["--noise_scale", str(self.noise_scale)])
        if self.noise_w is not None:
            command.extend(["--noise_w", str(self.noise_w)])
        return command


def normalize_language(language: str) -> str:
    normalized = (language or "en").strip().casefold()
    return normalized or "en"


def parse_voice_map_entries(entries: Optional[Sequence[str]]) -> Dict[str, str]:
    voice_map: Dict[str, str] = {}
    for entry in entries or ():
        if "=" not in entry:
            raise ValueError("voice map entries must use LANG=MODEL")
        language, model = entry.split("=", 1)
        language = normalize_language(language)
        model = model.strip()
        if not model:
            raise ValueError("voice map model must not be empty")
        voice_map[language] = model
    return voice_map


def read_wav_mono(path: Path) -> tuple:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        raw_audio = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        array = np.frombuffer(raw_audio, dtype=np.uint8).astype(np.float32)
        array = (array - 128.0) / 128.0
    elif sample_width == 2:
        array = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
        array = array / 32768.0
    elif sample_width == 4:
        array = np.frombuffer(raw_audio, dtype=np.int32).astype(np.float32)
        array = array / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")

    if channels > 1:
        array = array.reshape(-1, channels).mean(axis=1)

    return array.astype(np.float32), sample_rate


def mix_ambient(
    speech_samples: np.ndarray,
    sample_rate: int,
    ambient: AmbientMixConfig,
) -> np.ndarray:
    speech = np.asarray(speech_samples, dtype=np.float32) * ambient.speech_volume
    if ambient.audio_path is None:
        return np.clip(speech, -1.0, 1.0)

    ambient_samples, ambient_sample_rate = read_wav_mono(ambient.audio_path)
    if ambient_samples.size == 0 or speech.size == 0:
        return np.clip(speech, -1.0, 1.0)

    if ambient_sample_rate != sample_rate:
        ambient_samples = resample_linear(
            samples=ambient_samples,
            source_rate=ambient_sample_rate,
            target_rate=sample_rate,
        )

    repeated_ambient = loop_to_length(ambient_samples, speech.size)
    ducked_ambient = repeated_ambient * ambient.ambient_volume * ambient.ducking
    return np.clip(speech + ducked_ambient, -1.0, 1.0)


def loop_to_length(samples: np.ndarray, target_length: int) -> np.ndarray:
    if target_length <= 0:
        return np.array([], dtype=np.float32)
    if samples.size == 0:
        return np.zeros(target_length, dtype=np.float32)

    repeats = int(np.ceil(target_length / samples.size))
    return np.tile(samples, repeats)[:target_length].astype(np.float32)


def resample_linear(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be greater than 0")
    if source_rate == target_rate:
        return samples.astype(np.float32)
    if samples.size == 0:
        return samples.astype(np.float32)

    duration = samples.size / float(source_rate)
    target_size = max(1, int(round(duration * target_rate)))
    source_positions = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def play_samples(
    samples: np.ndarray,
    sample_rate: int,
    stop_event: threading.Event,
    block_seconds: float = 0.05,
) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        print("TTS playback error: sounddevice is not installed.")
        return

    block_size = max(1, int(sample_rate * block_seconds))
    position = 0
    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        while position < samples.size and not stop_event.is_set():
            block = samples[position : position + block_size]
            stream.write(block.reshape(-1, 1))
            position += block_size
