import unittest

from qloopbb.speech import speak_async


class SpeechTests(unittest.TestCase):
    def test_speak_async_disabled_returns_completed_handle(self) -> None:
        handle = speak_async("hello", enabled=False)

        self.assertIsNone(handle.thread)
        handle.wait()
        handle.stop()


if __name__ == "__main__":
    unittest.main()
