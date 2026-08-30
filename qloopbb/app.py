import argparse
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Tuple

from qloopbb.agent import build_reply
from qloopbb.asr import FasterWhisperTranscriber, TranscriptionResult
from qloopbb.audio import record_microphone, record_utterance, stream_utterance, write_wav
from qloopbb.chatter import build_chatter
from qloopbb.context import ContextResolution, resolve_route_context
from qloopbb.embeddings import DEFAULT_EMBEDDING_CACHE_DIR, DEFAULT_EMBEDDING_MODEL, SearchResult
from qloopbb.env import load_dotenv
from qloopbb.llm import (
    DEFAULT_GEMINI_MODEL,
    ConversationMemory,
    FixedReplyGenerator,
    GeminiGenerationError,
    GeminiRestReplyGenerator,
    ReplyContext,
    ReplyGenerator,
    get_gemini_api_key,
)
from qloopbb.localization import (
    GeminiOutputLocalizer,
    NoopOutputLocalizer,
    OutputLocalizationError,
    OutputLocalizer,
)
from qloopbb.retrieval import (
    LocalRetriever,
    normalize_query_text,
    load_retrieval_documents,
    print_search_results,
)
from qloopbb.router import RouteDecision, route_utterance
from qloopbb.tools import LocalHospitalTools, ToolResult
from qloopbb.tts import (
    AmbientMixConfig,
    NoopTtsEngine,
    PiperCliTtsEngine,
    TextToSpeechEngine,
    TtsHandle,
    parse_voice_map_entries,
)
from qloopbb.turns import build_turn_state


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def print_timing(enabled: bool, label: str, seconds: float) -> None:
    if enabled:
        print(f"Timing: {label}: {seconds * 1000:.1f} ms")


def print_timing_summary(enabled: bool, label: str, seconds: float) -> None:
    if enabled:
        print(f"Timing summary: {label}: {seconds * 1000:.1f} ms")


def print_route(route: RouteDecision) -> None:
    print(
        "Route: "
        f"{route.kind.value} "
        f"confidence={route.confidence:.2f} "
        f"retrieval={'yes' if route.needs_retrieval else 'no'} "
        f"tool={route.tool_name or 'none'}"
    )
    print(f"Route reason: {route.reason}")


def print_tool_result(tool_result: ToolResult) -> None:
    print(f"Tool result: {tool_result.name}")
    print(f"   {tool_result.summary}")
    for key, value in tool_result.data.items():
        print(f"   {key}: {value}")


def print_context_resolution(
    resolution: ContextResolution,
    route: RouteDecision,
    retriever: Optional[LocalRetriever],
    show_timings: bool,
) -> None:
    if resolution.tool_seconds is not None:
        print_timing(show_timings, "tool execution", resolution.tool_seconds)
    if resolution.tool_result is not None:
        print_tool_result(resolution.tool_result)

    if resolution.reused_prewarmed_results:
        print("Retrieval matches reused from prewarm.")
    elif resolution.retrieval_seconds is not None:
        print_timing(show_timings, "embedding retrieval", resolution.retrieval_seconds)
    elif retriever is not None:
        print(f"Retrieval skipped for route: {route.kind.value}")

    if resolution.retrieval_results is not None:
        print_search_results(resolution.retrieval_results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record speech, transcribe it locally, route it, and reply."
    )
    parser.add_argument(
        "--input-wav",
        type=Path,
        help="Use an existing WAV file instead of recording from the microphone.",
    )
    parser.add_argument(
        "--input-text",
        help="Use English text as the post-STT transcript for a full pipeline smoke test.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Microphone recording duration in seconds.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Keep listening for utterances until Ctrl+C.",
    )
    parser.add_argument(
        "--streaming-asr",
        action="store_true",
        help="Use rolling partial transcription and retrieval prewarming.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        help="Stop live mode after this many turns. Useful for testing.",
    )
    parser.add_argument(
        "--silence-seconds",
        type=float,
        default=0.8,
        help="Silence duration that ends a live utterance.",
    )
    parser.add_argument(
        "--max-utterance-seconds",
        type=float,
        default=20.0,
        help="Maximum duration for one live utterance.",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.015,
        help="Energy threshold for live microphone utterance detection.",
    )
    parser.add_argument(
        "--model-size",
        default="small",
        help="faster-whisper model size or local model path.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device passed to faster-whisper.",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Compute type passed to faster-whisper.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory used for downloaded faster-whisper models.",
    )
    parser.add_argument(
        "--language",
        help="Optional source language code, such as hi, ta, te, kn, ml, bn, mr, or en.",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Ask Whisper to translate non-English speech to English text.",
    )
    parser.add_argument(
        "--response-language",
        help="Optional output language override for future response translation and TTS.",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable faster-whisper VAD filtering.",
    )
    parser.add_argument(
        "--no-speak",
        action="store_true",
        help="Print replies without spoken audio.",
    )
    parser.add_argument(
        "--tts",
        choices=("auto", "console", "piper"),
        default="auto",
        help="Text-to-speech backend. Auto uses Piper when --tts-model is provided.",
    )
    parser.add_argument(
        "--tts-model",
        help="Default Piper model path or voice name.",
    )
    parser.add_argument(
        "--tts-voice-map",
        action="append",
        help="Language-specific Piper voice mapping in LANG=MODEL format.",
    )
    parser.add_argument(
        "--tts-piper-binary",
        default="piper",
        help="Piper executable path or command name.",
    )
    parser.add_argument(
        "--tts-speaker",
        help="Optional Piper speaker id or name for multi-speaker voices.",
    )
    parser.add_argument(
        "--tts-length-scale",
        type=float,
        help="Optional Piper length scale for speaking rate.",
    )
    parser.add_argument(
        "--tts-noise-scale",
        type=float,
        help="Optional Piper noise scale for voice variation.",
    )
    parser.add_argument(
        "--tts-noise-w",
        type=float,
        help="Optional Piper noise width for voice variation.",
    )
    parser.add_argument(
        "--ambient-wav",
        type=Path,
        help="Optional WAV file to mix quietly under generated TTS.",
    )
    parser.add_argument(
        "--ambient-volume",
        type=float,
        default=0.04,
        help="Background audio volume mixed under TTS.",
    )
    parser.add_argument(
        "--ambient-ducking",
        type=float,
        default=0.35,
        help="Background audio multiplier while speech is active.",
    )
    parser.add_argument(
        "--speech-volume",
        type=float,
        default=1.0,
        help="Generated speech volume before mixing.",
    )
    parser.add_argument(
        "--retrieve",
        action="store_true",
        help="Enable local vector retrieval for routes that need lookup context.",
    )
    parser.add_argument(
        "--retrieval-doc",
        action="append",
        dest="retrieval_docs",
        help="Document text to index for speech retrieval. Can be passed multiple times.",
    )
    parser.add_argument(
        "--retrieval-docs-file",
        type=Path,
        help="Optional newline-delimited text file of retrieval documents.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieval matches to print.",
    )
    parser.add_argument(
        "--embedding-model-name",
        default=DEFAULT_EMBEDDING_MODEL,
        help="fastembed model name for retrieval.",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_CACHE_DIR,
        help="Directory used for downloaded embedding models.",
    )
    parser.add_argument(
        "--embedding-threads",
        type=int,
        help="Optional number of embedding inference threads.",
    )
    parser.add_argument(
        "--timings",
        action="store_true",
        help="Print timing information for each pipeline stage.",
    )
    parser.add_argument(
        "--llm",
        choices=("fixed", "gemini"),
        default="fixed",
        help="Reply generator to use after routing, tools, and retrieval.",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model to use when --llm gemini is enabled.",
    )
    parser.add_argument(
        "--gemini-temperature",
        type=float,
        default=0.2,
        help="Gemini generation temperature.",
    )
    parser.add_argument(
        "--gemini-max-output-tokens",
        type=int,
        default=80,
        help="Maximum Gemini output tokens per reply.",
    )
    parser.add_argument(
        "--gemini-timeout-seconds",
        type=float,
        default=8.0,
        help="Gemini HTTP timeout per reply.",
    )
    parser.add_argument(
        "--output-localizer",
        choices=("auto", "none", "gemini"),
        default="auto",
        help="Translate English replies into the response language before TTS.",
    )
    parser.add_argument(
        "--localization-model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model to use when output localization uses Gemini.",
    )
    parser.add_argument(
        "--localization-max-output-tokens",
        type=int,
        default=120,
        help="Maximum Gemini output tokens per localized reply.",
    )
    parser.add_argument(
        "--localization-timeout-seconds",
        type=float,
        default=8.0,
        help="Gemini HTTP timeout per output localization request.",
    )
    parser.add_argument(
        "--conversation-turns",
        type=int,
        default=6,
        help="Number of previous turns to send to the reply generator.",
    )
    parser.add_argument(
        "--no-chatter",
        "--no-thinking-phrases",
        dest="no_chatter",
        action="store_true",
        help="Disable route-specific thinking phrases during tool or retrieval work.",
    )
    parser.add_argument(
        "--partial-interval-seconds",
        type=float,
        default=1.0,
        help="How often streaming ASR should attempt a partial transcript.",
    )
    parser.add_argument(
        "--partial-min-seconds",
        type=float,
        default=0.8,
        help="Minimum captured speech before the first partial transcript.",
    )
    parser.add_argument(
        "--partial-min-chars",
        type=int,
        default=8,
        help="Minimum partial transcript length needed before retrieval prewarming.",
    )
    return parser.parse_args()


def print_turn(
    transcriber: FasterWhisperTranscriber,
    audio_path: Path,
    language: Optional[str],
    translate_to_english: bool,
    whisper_vad: bool,
    speak_reply: bool,
    show_timings: bool,
    response_language_override: Optional[str] = None,
    retriever: Optional[LocalRetriever] = None,
    tools: Optional[LocalHospitalTools] = None,
    reply_generator: Optional[ReplyGenerator] = None,
    conversation: Optional[ConversationMemory] = None,
    output_localizer: Optional[OutputLocalizer] = None,
    tts_engine: Optional[TextToSpeechEngine] = None,
    chatter_enabled: bool = True,
    top_k: int = 3,
) -> None:
    turn_start = perf_counter()
    transcribe_start = perf_counter()
    result = transcriber.transcribe(
        audio_path=audio_path,
        language=language,
        translate_to_english=translate_to_english,
        vad_filter=whisper_vad,
    )
    print_timing(show_timings, "whisper transcription", perf_counter() - transcribe_start)

    state_start = perf_counter()
    turn_state = build_turn_state(
        transcription=result,
        translated_to_english=translate_to_english,
        response_language_override=response_language_override,
    )
    print_timing(show_timings, "turn state", perf_counter() - state_start)

    print(f"Detected language: {turn_state.source_language or 'unknown'}")
    print(f"Working language: {turn_state.working_language}")
    print(f"Response language: {turn_state.response_language}")
    transcript_label = "English transcript" if translate_to_english else "Transcript"
    print(f"{transcript_label}: {turn_state.working_text or '[no speech detected]'}")

    route_start = perf_counter()
    route = route_utterance(turn_state.working_text)
    print_timing(show_timings, "routing", perf_counter() - route_start)
    print_route(route)

    context_resolution, chatter_handle = resolve_context_with_chatter(
        route=route,
        transcript=turn_state.working_text,
        tools=tools,
        retriever=retriever,
        top_k=top_k,
        chatter_enabled=chatter_enabled,
        output_localizer=output_localizer or NoopOutputLocalizer(),
        tts_engine=tts_engine or NoopTtsEngine(),
        language=turn_state.response_language,
        show_timings=show_timings,
    )
    print_context_resolution(
        resolution=context_resolution,
        route=route,
        retriever=retriever,
        show_timings=show_timings,
    )

    reply_start = perf_counter()
    reply = generate_reply(
        reply_generator=reply_generator or FixedReplyGenerator(),
        conversation=conversation or ConversationMemory(max_turns=0),
        context=ReplyContext(
            transcript=turn_state.working_text,
            source_language=turn_state.source_language,
            response_language=turn_state.response_language,
            route=route,
            tool_result=context_resolution.tool_result,
            retrieval_results=context_resolution.retrieval_results,
            history=(conversation.recent() if conversation is not None else ()),
        ),
    )
    print_timing(show_timings, "reply generation", perf_counter() - reply_start)
    print(f"English reply: {reply}")

    localized_reply = localize_for_output(
        output_localizer=output_localizer or NoopOutputLocalizer(),
        text=reply,
        target_language=turn_state.response_language,
        show_timings=show_timings,
        timing_label="reply localization",
    )
    print_localized_text("Localized reply", localized_reply, reply, turn_state.response_language)

    wait_for_chatter(chatter_handle, show_timings)

    speech_start = perf_counter()
    if speak_reply:
        (tts_engine or NoopTtsEngine()).speak(
            localized_reply,
            language=turn_state.response_language,
        )
    print_timing(show_timings, "speech output", perf_counter() - speech_start)
    print_timing_summary(
        show_timings,
        "warm speech roundtrip after capture",
        perf_counter() - turn_start,
    )


class StaticTextTranscriber:
    def __init__(self, text: str, language: str = "en") -> None:
        self.text = text
        self.language = language

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str],
        translate_to_english: bool,
        vad_filter: bool,
    ) -> TranscriptionResult:
        return TranscriptionResult(
            text=self.text,
            language=language or self.language,
            duration_seconds=0.0,
        )


def build_transcriber(args: argparse.Namespace) -> FasterWhisperTranscriber:
    print("Loading transcription model...")
    start = perf_counter()
    transcriber = FasterWhisperTranscriber(
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        download_root=args.model_dir,
    )
    print_timing(args.timings, "whisper model load", perf_counter() - start)
    return transcriber


def build_retriever(args: argparse.Namespace) -> Optional[LocalRetriever]:
    if not args.retrieve:
        return None

    documents = load_retrieval_documents(
        docs=args.retrieval_docs,
        docs_file=args.retrieval_docs_file,
    )
    return LocalRetriever(
        documents=documents,
        model_name=args.embedding_model_name,
        cache_dir=args.embedding_cache_dir,
        threads=args.embedding_threads,
        show_timings=args.timings,
    )


def build_reply_generator(args: argparse.Namespace) -> ReplyGenerator:
    if args.llm == "fixed":
        return FixedReplyGenerator()

    print(f"Using Gemini model: {args.gemini_model}")
    return GeminiRestReplyGenerator(
        model=args.gemini_model,
        temperature=args.gemini_temperature,
        max_output_tokens=args.gemini_max_output_tokens,
        timeout_seconds=args.gemini_timeout_seconds,
    )


def build_output_localizer(args: argparse.Namespace) -> OutputLocalizer:
    if args.output_localizer == "none":
        return NoopOutputLocalizer()
    if args.output_localizer == "auto" and args.llm != "gemini":
        return NoopOutputLocalizer()

    print(f"Using output localizer: gemini {args.localization_model}")
    return GeminiOutputLocalizer(
        model=args.localization_model,
        max_output_tokens=args.localization_max_output_tokens,
        timeout_seconds=args.localization_timeout_seconds,
    )


def build_tts_engine(args: argparse.Namespace) -> TextToSpeechEngine:
    if args.no_speak or args.tts == "console":
        return NoopTtsEngine()
    if args.tts == "auto" and not args.tts_model:
        return NoopTtsEngine()
    if not args.tts_model:
        raise SystemExit("--tts piper requires --tts-model.")

    try:
        voice_map = parse_voice_map_entries(args.tts_voice_map)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print("Using TTS engine: piper")
    return PiperCliTtsEngine(
        model=args.tts_model,
        binary=args.tts_piper_binary,
        model_by_language=voice_map,
        speaker=args.tts_speaker,
        length_scale=args.tts_length_scale,
        noise_scale=args.tts_noise_scale,
        noise_w=args.tts_noise_w,
        ambient=AmbientMixConfig(
            audio_path=args.ambient_wav,
            ambient_volume=args.ambient_volume,
            speech_volume=args.speech_volume,
            ducking=args.ambient_ducking,
        ),
    )


def generate_reply(
    reply_generator: ReplyGenerator,
    conversation: ConversationMemory,
    context: ReplyContext,
) -> str:
    try:
        reply = reply_generator.generate(context)
    except GeminiGenerationError as exc:
        print(f"LLM error: {exc}")
        reply = build_reply(
            context.transcript,
            route=context.route,
            tool_result=context.tool_result,
            retrieval_results=list(context.retrieval_results or []),
        )

    conversation.append(context.transcript, reply)
    return reply


def localize_for_output(
    output_localizer: OutputLocalizer,
    text: str,
    target_language: str,
    show_timings: bool,
    timing_label: str,
) -> str:
    localization_start = perf_counter()
    try:
        localized_text = output_localizer.localize(text, target_language)
    except OutputLocalizationError as exc:
        print(f"Localization error: {exc}")
        localized_text = text
    print_timing(show_timings, timing_label, perf_counter() - localization_start)
    return localized_text


def print_localized_text(
    label: str,
    localized_text: str,
    english_text: str,
    target_language: str,
) -> None:
    if localized_text != english_text:
        print(f"{label} ({target_language}): {localized_text}")


def should_translate_to_english(
    args: argparse.Namespace,
    retriever: Optional[LocalRetriever],
) -> bool:
    return (
        args.translate
        or retriever is not None
        or args.llm == "gemini"
        or args.output_localizer == "gemini"
    )


def validate_runtime_settings(args: argparse.Namespace) -> None:
    if args.input_text and (args.live or args.streaming_asr):
        raise SystemExit("--input-text cannot be combined with --live or --streaming-asr.")
    if args.input_text and args.input_wav:
        raise SystemExit("--input-text cannot be combined with --input-wav.")
    if args.conversation_turns < 0:
        raise SystemExit("--conversation-turns must be 0 or greater.")
    if args.llm == "gemini" and not get_gemini_api_key():
        raise SystemExit("Set GEMINI_API_KEY before using --llm gemini.")
    if args.output_localizer == "gemini" and not get_gemini_api_key():
        raise SystemExit("Set GEMINI_API_KEY before using --output-localizer gemini.")
    if args.tts == "piper" and not args.no_speak and not args.tts_model:
        raise SystemExit("--tts piper requires --tts-model.")
    if args.ambient_wav is not None and not args.ambient_wav.exists():
        raise SystemExit(f"--ambient-wav does not exist: {args.ambient_wav}")
    validate_non_negative("--ambient-volume", args.ambient_volume)
    validate_non_negative("--ambient-ducking", args.ambient_ducking)
    validate_non_negative("--speech-volume", args.speech_volume)


def validate_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise SystemExit(f"{name} must be 0 or greater.")


def resolve_context_with_chatter(
    route: RouteDecision,
    transcript: str,
    tools: Optional[LocalHospitalTools],
    retriever: Optional[LocalRetriever],
    top_k: int,
    chatter_enabled: bool,
    output_localizer: OutputLocalizer,
    tts_engine: TextToSpeechEngine,
    language: str,
    show_timings: bool,
    prewarmed_results: Optional[List[SearchResult]] = None,
    prewarmed_query_key: str = "",
) -> Tuple[ContextResolution, Optional[TtsHandle]]:
    chatter = build_chatter(route)
    should_chatter = (
        chatter_enabled
        and chatter is not None
        and context_work_needed(
            route=route,
            transcript=transcript,
            tools=tools,
            retriever=retriever,
        )
    )

    if not should_chatter:
        return (
            resolve_route_context(
                route=route,
                transcript=transcript,
                tools=tools,
                retriever=retriever,
                top_k=top_k,
                prewarmed_results=prewarmed_results,
                prewarmed_query_key=prewarmed_query_key,
            ),
            None,
        )

    context_start = perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            resolve_route_context,
            route,
            transcript,
            tools,
            retriever,
            top_k,
            prewarmed_results,
            prewarmed_query_key,
        )
        localized_chatter = localize_for_output(
            output_localizer=output_localizer,
            text=chatter,
            target_language=language,
            show_timings=show_timings,
            timing_label="thinking phrase localization",
        )
        print(f"Thinking phrase: {chatter}")
        print_localized_text("Localized thinking phrase", localized_chatter, chatter, language)
        chatter_handle = tts_engine.speak_async(
            localized_chatter,
            language=language,
            is_thinking=True,
        )
        resolution = future.result()
    print_timing(show_timings, "background context resolution", perf_counter() - context_start)
    return resolution, chatter_handle


def context_work_needed(
    route: RouteDecision,
    transcript: str,
    tools: Optional[LocalHospitalTools],
    retriever: Optional[LocalRetriever],
) -> bool:
    return (tools is not None and route.tool_name is not None) or (
        retriever is not None and route.needs_retrieval and bool(transcript)
    )


def wait_for_chatter(
    chatter_handle: Optional[TtsHandle],
    show_timings: bool,
) -> None:
    if chatter_handle is None:
        return

    chatter_wait_start = perf_counter()
    chatter_handle.wait()
    print_timing(show_timings, "chatter completion wait", perf_counter() - chatter_wait_start)


def run_streaming_live(args: argparse.Namespace) -> None:
    reply_generator = build_reply_generator(args)
    output_localizer = build_output_localizer(args)
    conversation = ConversationMemory(max_turns=args.conversation_turns)
    tts_engine = build_tts_engine(args)
    transcriber = build_transcriber(args)
    retriever = build_retriever(args)
    tools = LocalHospitalTools()
    translate_to_english = True
    turns = 0

    with tempfile.TemporaryDirectory(prefix="qloopbb-stream-") as temp_dir:
        temp_path = Path(temp_dir)
        print("Streaming ASR mode started. Press Ctrl+C to stop.")
        print("Streaming mode uses English transcripts for downstream search.")
        try:
            while args.max_turns is None or turns < args.max_turns:
                run_streaming_turn(
                    args=args,
                    transcriber=transcriber,
                    retriever=retriever,
                    tools=tools,
                    reply_generator=reply_generator,
                    conversation=conversation,
                    output_localizer=output_localizer,
                    tts_engine=tts_engine,
                    temp_path=temp_path,
                    turn_number=turns + 1,
                    translate_to_english=translate_to_english,
                )
                turns += 1
                print()
        except KeyboardInterrupt:
            print("\nStreaming ASR mode stopped.")


def run_streaming_turn(
    args: argparse.Namespace,
    transcriber: FasterWhisperTranscriber,
    retriever: Optional[LocalRetriever],
    tools: LocalHospitalTools,
    reply_generator: ReplyGenerator,
    conversation: ConversationMemory,
    output_localizer: OutputLocalizer,
    tts_engine: TextToSpeechEngine,
    temp_path: Path,
    turn_number: int,
    translate_to_english: bool,
) -> None:
    final_event = None
    last_partial_text = ""
    last_partial_key = ""
    prewarmed_results = None
    first_partial_at = None
    first_prewarm_at = None
    endpoint_to_reply_start = None

    for partial_number, event in enumerate(
        stream_utterance(
            silence_seconds=args.silence_seconds,
            max_seconds=args.max_utterance_seconds,
            vad_threshold=args.vad_threshold,
            partial_interval_seconds=args.partial_interval_seconds,
            partial_min_seconds=args.partial_min_seconds,
        ),
        start=1,
    ):
        audio_path = temp_path / f"turn-{turn_number}-partial-{partial_number}.wav"
        if event.is_final:
            audio_path = temp_path / f"turn-{turn_number}-final.wav"
            final_event = event
        write_wav(audio_path, event.samples)

        if event.is_final:
            break

        partial_start = perf_counter()
        partial_result = transcriber.transcribe(
            audio_path=audio_path,
            language=args.language,
            translate_to_english=translate_to_english,
            vad_filter=False,
        )
        partial_seconds = perf_counter() - partial_start
        partial_text = partial_result.text.strip()
        if not partial_text or partial_text == last_partial_text:
            continue

        if first_partial_at is None:
            first_partial_at = event.elapsed_seconds + partial_seconds
        last_partial_text = partial_text
        last_partial_key = normalize_query_text(partial_text)
        print(f"Partial English transcript: {partial_text}")
        print_timing(args.timings, "partial whisper transcription", partial_seconds)
        print_timing_summary(
            args.timings,
            "first partial transcript after speech start",
            first_partial_at,
        )

        partial_route = route_utterance(partial_text)
        should_prewarm_retrieval = (
            retriever is not None
            and partial_route.needs_retrieval
            and len(partial_text) >= args.partial_min_chars
        )
        if should_prewarm_retrieval:
            prewarm_start = perf_counter()
            prewarmed_results = retriever.search(partial_text, top_k=args.top_k)
            prewarm_seconds = perf_counter() - prewarm_start
            if first_prewarm_at is None:
                first_prewarm_at = event.elapsed_seconds + partial_seconds + prewarm_seconds
            print_timing(args.timings, "partial retrieval prewarm", prewarm_seconds)
            print_timing_summary(
                args.timings,
                "first retrieval prewarm after speech start",
                first_prewarm_at,
            )
            if prewarmed_results:
                top = prewarmed_results[0]
                print(f"Prewarm top match: score={top.score:.4f} id={top.document.id}")
                print(f"   {top.document.text}")

    if final_event is None:
        return

    endpoint_start = perf_counter()
    final_audio_path = temp_path / f"turn-{turn_number}-final.wav"
    final_transcribe_start = perf_counter()
    final_result = transcriber.transcribe(
        audio_path=final_audio_path,
        language=args.language,
        translate_to_english=translate_to_english,
        vad_filter=False,
    )
    print_timing(
        args.timings,
        "final whisper transcription",
        perf_counter() - final_transcribe_start,
    )

    turn_state = build_turn_state(
        transcription=final_result,
        translated_to_english=translate_to_english,
        response_language_override=args.response_language,
    )
    print(f"Detected language: {turn_state.source_language or 'unknown'}")
    print(f"Working language: {turn_state.working_language}")
    print(f"Response language: {turn_state.response_language}")
    print(f"Final English transcript: {turn_state.working_text or '[no speech detected]'}")

    route_start = perf_counter()
    route = route_utterance(turn_state.working_text)
    print_timing(args.timings, "routing", perf_counter() - route_start)
    print_route(route)

    context_resolution, chatter_handle = resolve_context_with_chatter(
        route=route,
        transcript=turn_state.working_text,
        tools=tools,
        retriever=retriever,
        top_k=args.top_k,
        chatter_enabled=not args.no_chatter,
        output_localizer=output_localizer,
        tts_engine=tts_engine,
        language=turn_state.response_language,
        show_timings=args.timings,
        prewarmed_results=prewarmed_results,
        prewarmed_query_key=last_partial_key,
    )
    print_context_resolution(
        resolution=context_resolution,
        route=route,
        retriever=retriever,
        show_timings=args.timings,
    )

    reply_start = perf_counter()
    reply = generate_reply(
        reply_generator=reply_generator,
        conversation=conversation,
        context=ReplyContext(
            transcript=turn_state.working_text,
            source_language=turn_state.source_language,
            response_language=turn_state.response_language,
            route=route,
            tool_result=context_resolution.tool_result,
            retrieval_results=context_resolution.retrieval_results,
            history=conversation.recent(),
        ),
    )
    print_timing(args.timings, "reply generation", perf_counter() - reply_start)
    print(f"English reply: {reply}")

    localized_reply = localize_for_output(
        output_localizer=output_localizer,
        text=reply,
        target_language=turn_state.response_language,
        show_timings=args.timings,
        timing_label="reply localization",
    )
    print_localized_text("Localized reply", localized_reply, reply, turn_state.response_language)

    wait_for_chatter(chatter_handle, args.timings)

    speech_start = perf_counter()
    tts_engine.speak(localized_reply, language=turn_state.response_language)
    print_timing(args.timings, "speech output", perf_counter() - speech_start)

    endpoint_to_reply_start = perf_counter() - endpoint_start
    print_timing_summary(
        args.timings,
        "streaming warm roundtrip after endpoint",
        endpoint_to_reply_start,
    )


def run_live(args: argparse.Namespace) -> None:
    reply_generator = build_reply_generator(args)
    output_localizer = build_output_localizer(args)
    conversation = ConversationMemory(max_turns=args.conversation_turns)
    tts_engine = build_tts_engine(args)
    transcriber = build_transcriber(args)
    retriever = build_retriever(args)
    tools = LocalHospitalTools()
    translate_to_english = should_translate_to_english(args, retriever)
    turns = 0

    with tempfile.TemporaryDirectory(prefix="qloopbb-live-") as temp_dir:
        temp_path = Path(temp_dir)
        print("Live mode started. Press Ctrl+C to stop.")
        if retriever is not None:
            print("Retrieval mode uses English transcripts for downstream search.")
        try:
            while args.max_turns is None or turns < args.max_turns:
                audio_path = temp_path / f"turn-{turns + 1}.wav"
                listen_start = perf_counter()
                record_utterance(
                    output_path=audio_path,
                    silence_seconds=args.silence_seconds,
                    max_seconds=args.max_utterance_seconds,
                    vad_threshold=args.vad_threshold,
                )
                print_timing(args.timings, "live listen and capture", perf_counter() - listen_start)
                print_turn(
                    transcriber=transcriber,
                    audio_path=audio_path,
                    language=args.language,
                    translate_to_english=translate_to_english,
                    whisper_vad=not args.no_vad,
                    speak_reply=not args.no_speak,
                    show_timings=args.timings,
                    response_language_override=args.response_language,
                    retriever=retriever,
                    tools=tools,
                    reply_generator=reply_generator,
                    conversation=conversation,
                    output_localizer=output_localizer,
                    tts_engine=tts_engine,
                    chatter_enabled=not args.no_chatter,
                    top_k=args.top_k,
                )
                turns += 1
                print()
        except KeyboardInterrupt:
            print("\nLive mode stopped.")


def run_once(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(prefix="qloopbb-") as temp_dir:
        if args.input_text is not None:
            reply_generator = build_reply_generator(args)
            output_localizer = build_output_localizer(args)
            conversation = ConversationMemory(max_turns=args.conversation_turns)
            tts_engine = build_tts_engine(args)
            retriever = build_retriever(args)
            tools = LocalHospitalTools()
            print_turn(
                transcriber=StaticTextTranscriber(
                    text=args.input_text,
                    language=args.language or "en",
                ),
                audio_path=Path("<input-text>"),
                language=args.language or "en",
                translate_to_english=True,
                whisper_vad=False,
                speak_reply=not args.no_speak,
                show_timings=args.timings,
                response_language_override=args.response_language,
                retriever=retriever,
                tools=tools,
                reply_generator=reply_generator,
                conversation=conversation,
                output_localizer=output_localizer,
                tts_engine=tts_engine,
                chatter_enabled=not args.no_chatter,
                top_k=args.top_k,
            )
            return

        audio_path = args.input_wav
        if audio_path is None:
            audio_path = Path(temp_dir) / "input.wav"
            record_start = perf_counter()
            record_microphone(args.duration, audio_path)
            print_timing(args.timings, "fixed-duration recording", perf_counter() - record_start)

        reply_generator = build_reply_generator(args)
        output_localizer = build_output_localizer(args)
        conversation = ConversationMemory(max_turns=args.conversation_turns)
        tts_engine = build_tts_engine(args)
        transcriber = build_transcriber(args)
        retriever = build_retriever(args)
        tools = LocalHospitalTools()
        translate_to_english = should_translate_to_english(args, retriever)
        print_turn(
            transcriber=transcriber,
            audio_path=audio_path,
            language=args.language,
            translate_to_english=translate_to_english,
            whisper_vad=not args.no_vad,
            speak_reply=not args.no_speak,
            show_timings=args.timings,
            response_language_override=args.response_language,
            retriever=retriever,
            tools=tools,
            reply_generator=reply_generator,
            conversation=conversation,
            output_localizer=output_localizer,
            tts_engine=tts_engine,
            chatter_enabled=not args.no_chatter,
            top_k=args.top_k,
        )


def main() -> None:
    load_dotenv()
    args = parse_args()
    validate_runtime_settings(args)
    if args.streaming_asr:
        run_streaming_live(args)
    elif args.live:
        run_live(args)
    else:
        run_once(args)


if __name__ == "__main__":
    main()
