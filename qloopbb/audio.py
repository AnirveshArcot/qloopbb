import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
from time import perf_counter
from typing import Deque, Iterator, List, Optional, Tuple

import numpy as np


SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class UtteranceAudioEvent:
    samples: np.ndarray
    elapsed_seconds: float
    is_final: bool


def record_microphone(duration_seconds: float, output_path: Path) -> Path:
    """Record mono 16 kHz microphone audio to a WAV file."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is not installed. Run `python -m pip install -r requirements.txt`."
        ) from exc

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than 0")

    frames = int(duration_seconds * SAMPLE_RATE)
    print(f"Recording for {duration_seconds:g}s...")
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    write_wav(output_path, audio[:, 0], SAMPLE_RATE)
    return output_path


def record_utterance(
    output_path: Path,
    silence_seconds: float = 0.8,
    max_seconds: float = 20.0,
    min_speech_seconds: float = 0.25,
    vad_threshold: float = 0.015,
    block_seconds: float = 0.1,
    pre_roll_seconds: float = 0.3,
) -> Path:
    """Record one utterance using a lightweight energy VAD."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is not installed. Run `python -m pip install -r requirements.txt`."
        ) from exc

    if silence_seconds <= 0:
        raise ValueError("silence_seconds must be greater than 0")
    if max_seconds <= 0:
        raise ValueError("max_seconds must be greater than 0")
    if min_speech_seconds < 0:
        raise ValueError("min_speech_seconds must be 0 or greater")
    if vad_threshold <= 0:
        raise ValueError("vad_threshold must be greater than 0")
    if block_seconds <= 0:
        raise ValueError("block_seconds must be greater than 0")

    block_frames = max(1, int(SAMPLE_RATE * block_seconds))
    silence_blocks_needed = max(1, int(silence_seconds / block_seconds))
    max_blocks = max(1, int(max_seconds / block_seconds))
    min_speech_blocks = max(1, int(min_speech_seconds / block_seconds))
    pre_roll_blocks = max(0, int(pre_roll_seconds / block_seconds))
    pre_roll: Deque[np.ndarray] = deque(maxlen=pre_roll_blocks)
    captured: List[np.ndarray] = []
    speech_blocks = 0
    silent_blocks = 0
    started = False

    print("Listening...")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=block_frames,
    ) as stream:
        while True:
            block, overflowed = stream.read(block_frames)
            if overflowed:
                print("Audio input overflowed; continuing.")

            mono_block = block[:, 0].copy()
            rms = float(np.sqrt(np.mean(np.square(mono_block))))
            is_speech = rms >= vad_threshold

            if not started:
                if not is_speech:
                    pre_roll.append(mono_block)
                    continue
                print("Speech detected.")
                started = True
                captured.extend(pre_roll)
                pre_roll.clear()

            captured.append(mono_block)
            if is_speech:
                speech_blocks += 1
                silent_blocks = 0
            else:
                silent_blocks += 1

            enough_speech = speech_blocks >= min_speech_blocks
            enough_silence = silent_blocks >= silence_blocks_needed
            hit_max_length = len(captured) >= max_blocks
            if enough_speech and (enough_silence or hit_max_length):
                break

    samples = np.concatenate(captured) if captured else np.array([], dtype=np.float32)
    write_wav(output_path, samples, SAMPLE_RATE)
    return output_path


def stream_utterance(
    silence_seconds: float = 0.8,
    max_seconds: float = 20.0,
    min_speech_seconds: float = 0.25,
    vad_threshold: float = 0.015,
    block_seconds: float = 0.1,
    pre_roll_seconds: float = 0.3,
    partial_interval_seconds: float = 1.0,
    partial_min_seconds: float = 0.8,
) -> Iterator[UtteranceAudioEvent]:
    """Yield partial and final utterance audio while recording continues."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is not installed. Run `python -m pip install -r requirements.txt`."
        ) from exc

    validate_live_audio_settings(
        silence_seconds=silence_seconds,
        max_seconds=max_seconds,
        min_speech_seconds=min_speech_seconds,
        vad_threshold=vad_threshold,
        block_seconds=block_seconds,
    )
    if partial_interval_seconds <= 0:
        raise ValueError("partial_interval_seconds must be greater than 0")
    if partial_min_seconds <= 0:
        raise ValueError("partial_min_seconds must be greater than 0")

    block_frames = max(1, int(SAMPLE_RATE * block_seconds))
    silence_blocks_needed = max(1, int(silence_seconds / block_seconds))
    max_blocks = max(1, int(max_seconds / block_seconds))
    min_speech_blocks = max(1, int(min_speech_seconds / block_seconds))
    pre_roll_blocks = max(0, int(pre_roll_seconds / block_seconds))
    partial_blocks_needed = max(1, int(partial_interval_seconds / block_seconds))
    partial_min_blocks = max(1, int(partial_min_seconds / block_seconds))

    pre_roll: Deque[np.ndarray] = deque(maxlen=pre_roll_blocks)
    captured: List[np.ndarray] = []
    speech_blocks = 0
    silent_blocks = 0
    started_at: Optional[float] = None
    blocks_since_partial = 0
    audio_blocks: Queue[Tuple[np.ndarray, bool]] = Queue(maxsize=max_blocks + 50)

    def on_audio(indata: np.ndarray, _frames: int, _time_info: object, status: object) -> None:
        overflowed = bool(status)
        try:
            audio_blocks.put_nowait((indata[:, 0].copy(), overflowed))
        except Full:
            pass

    print("Listening...")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=block_frames,
        callback=on_audio,
    ):
        while True:
            mono_block, overflowed = audio_blocks.get()
            if overflowed:
                print("Audio input overflowed; continuing.")

            rms = float(np.sqrt(np.mean(np.square(mono_block))))
            is_speech = rms >= vad_threshold

            if started_at is None:
                if not is_speech:
                    pre_roll.append(mono_block)
                    continue
                print("Speech detected.")
                started_at = perf_counter()
                captured.extend(pre_roll)
                pre_roll.clear()

            captured.append(mono_block)
            blocks_since_partial += 1

            if is_speech:
                speech_blocks += 1
                silent_blocks = 0
            else:
                silent_blocks += 1

            elapsed_seconds = perf_counter() - started_at
            enough_partial_audio = len(captured) >= partial_min_blocks
            enough_partial_interval = blocks_since_partial >= partial_blocks_needed
            if enough_partial_audio and enough_partial_interval:
                blocks_since_partial = 0
                yield UtteranceAudioEvent(
                    samples=np.concatenate(captured),
                    elapsed_seconds=elapsed_seconds,
                    is_final=False,
                )

            enough_speech = speech_blocks >= min_speech_blocks
            enough_silence = silent_blocks >= silence_blocks_needed
            hit_max_length = len(captured) >= max_blocks
            if enough_speech and (enough_silence or hit_max_length):
                yield UtteranceAudioEvent(
                    samples=np.concatenate(captured),
                    elapsed_seconds=elapsed_seconds,
                    is_final=True,
                )
                break


def validate_live_audio_settings(
    silence_seconds: float,
    max_seconds: float,
    min_speech_seconds: float,
    vad_threshold: float,
    block_seconds: float,
) -> None:
    if silence_seconds <= 0:
        raise ValueError("silence_seconds must be greater than 0")
    if max_seconds <= 0:
        raise ValueError("max_seconds must be greater than 0")
    if min_speech_seconds < 0:
        raise ValueError("min_speech_seconds must be 0 or greater")
    if vad_threshold <= 0:
        raise ValueError("vad_threshold must be greater than 0")
    if block_seconds <= 0:
        raise ValueError("block_seconds must be greater than 0")


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write float samples in [-1.0, 1.0] to a 16-bit PCM WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
