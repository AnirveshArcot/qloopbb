import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from qloopbb.tts import (
    AmbientMixConfig,
    NoopTtsEngine,
    PiperCliTtsEngine,
    loop_to_length,
    mix_ambient,
    parse_voice_map_entries,
    read_wav_mono,
    resample_linear,
)


class TtsTests(unittest.TestCase):
    def test_noop_tts_returns_completed_handle(self) -> None:
        handle = NoopTtsEngine().speak_async("hello")

        self.assertIsNone(handle.thread)
        handle.wait()
        handle.stop()

    def test_parse_voice_map_entries(self) -> None:
        parsed = parse_voice_map_entries(
            [
                "hi=models/tts/hindi.onnx",
                "ta=models/tts/tamil.onnx",
            ]
        )

        self.assertEqual(parsed["hi"], "models/tts/hindi.onnx")
        self.assertEqual(parsed["ta"], "models/tts/tamil.onnx")

    def test_piper_selects_language_specific_model(self) -> None:
        engine = PiperCliTtsEngine(
            model="models/tts/en.onnx",
            model_by_language={
                "hi": "models/tts/hi.onnx",
                "ta-IN": "models/tts/ta.onnx",
            },
        )

        self.assertEqual(engine.select_model("hi-IN"), "models/tts/hi.onnx")
        self.assertEqual(engine.select_model("ta-in"), "models/tts/ta.onnx")
        self.assertEqual(engine.select_model("en"), "models/tts/en.onnx")

    def test_piper_command_includes_voice_controls(self) -> None:
        engine = PiperCliTtsEngine(
            model="models/tts/en.onnx",
            binary="piper",
            speaker="2",
            length_scale=0.9,
            noise_scale=0.5,
            noise_w=0.7,
        )

        command = engine._build_command(
            model="models/tts/en.onnx",
            output_path=Path("/tmp/out.wav"),
        )

        self.assertIn("--speaker", command)
        self.assertIn("--length_scale", command)
        self.assertIn("--noise_scale", command)
        self.assertIn("--noise_w", command)

    def test_loop_to_length_repeats_audio(self) -> None:
        samples = np.array([0.1, 0.2], dtype=np.float32)

        looped = loop_to_length(samples, 5)

        np.testing.assert_allclose(
            looped,
            np.array([0.1, 0.2, 0.1, 0.2, 0.1], dtype=np.float32),
        )

    def test_resample_linear_changes_length(self) -> None:
        samples = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)

        resampled = resample_linear(samples, source_rate=4, target_rate=8)

        self.assertEqual(resampled.size, 8)

    def test_mix_ambient_adds_ducked_background(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ambient_path = Path(temp_dir) / "ambient.wav"
            write_test_wav(
                ambient_path,
                np.array([0.5, -0.5], dtype=np.float32),
                sample_rate=4,
            )
            speech = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)

            mixed = mix_ambient(
                speech,
                sample_rate=4,
                ambient=AmbientMixConfig(
                    audio_path=ambient_path,
                    ambient_volume=0.1,
                    ducking=0.5,
                    speech_volume=1.0,
                ),
            )

            np.testing.assert_allclose(
                mixed,
                np.array([0.225, 0.175, 0.225, 0.175], dtype=np.float32),
                rtol=1e-4,
                atol=1e-4,
            )

    def test_read_wav_mono_averages_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stereo.wav"
            stereo = np.array(
                [
                    [0.5, -0.5],
                    [0.25, 0.25],
                ],
                dtype=np.float32,
            )
            write_test_wav(path, stereo, sample_rate=8)

            samples, sample_rate = read_wav_mono(path)

            self.assertEqual(sample_rate, 8)
            np.testing.assert_allclose(
                samples,
                np.array([0.0, 0.25], dtype=np.float32),
                rtol=1e-4,
                atol=1e-4,
            )


def write_test_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    channels = 1 if pcm.ndim == 1 else pcm.shape[1]

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


if __name__ == "__main__":
    unittest.main()
