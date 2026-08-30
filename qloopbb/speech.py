from qloopbb.tts import NoopTtsEngine, TtsHandle


def speak(text: str, enabled: bool = True) -> None:
    """Compatibility wrapper for older call sites."""
    speak_async(text=text, enabled=enabled).wait()


def speak_async(text: str, enabled: bool = True) -> TtsHandle:
    """Compatibility wrapper for older call sites."""
    if enabled:
        print(text)
    return NoopTtsEngine().speak_async(text=text)
