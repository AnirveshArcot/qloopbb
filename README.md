# QloopBB

Local-first hospital voice agent prototype.

## Current Prototype

This prototype records speech, transcribes it locally with `faster-whisper`, routes the English working transcript, optionally runs retrieval or local administrative tools, and then replies.
Small talk still falls back to:

```text
that's cool
```

The goal is to prove the local speech, routing, retrieval, and tool boundaries before adding a local LLM brain.
For now, Gemini can be used as a temporary hosted reply generator and output localizer while the local LLM and local translation paths are still being worked out.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first transcription run downloads the selected Whisper model if it is not already cached.
By default, models are stored in the ignored `models/` directory.

## Run With Microphone

```bash
python -m qloopbb.app --duration 5
```

By default, the app prints replies and does not speak unless a TTS backend is configured.
Use `--no-speak` to force print-only mode.

```bash
python -m qloopbb.app --duration 5 --no-speak
```

## Run With Piper TTS

The app no longer uses macOS `say`.
For local neural speech, install a Piper executable and provide a voice model:

```bash
python -m qloopbb.app --live --tts piper --tts-model models/tts/en_US-lessac-medium.onnx --model-size tiny --model-dir models
```

Piper voices are separate assets.
Each voice normally has a `.onnx` model and a matching `.onnx.json` config file.

Map response languages to different Piper voices:

```bash
python -m qloopbb.app \
  --live \
  --retrieve \
  --tts piper \
  --tts-model models/tts/en_US-lessac-medium.onnx \
  --tts-voice-map hi=models/tts/hi_IN-example-medium.onnx \
  --tts-voice-map ta=models/tts/ta_IN-example-medium.onnx \
  --model-size tiny \
  --model-dir models
```

When output localization is enabled, the app translates the English reply before TTS.
The TTS layer uses response-language state to select the right voice.

Voice and playback controls:

```bash
python -m qloopbb.app \
  --live \
  --tts piper \
  --tts-model models/tts/en_US-lessac-medium.onnx \
  --tts-length-scale 0.95 \
  --tts-noise-scale 0.5 \
  --speech-volume 0.9 \
  --model-size tiny \
  --model-dir models
```

Add an ambient WAV track under generated speech:

```bash
python -m qloopbb.app \
  --live \
  --tts piper \
  --tts-model models/tts/en_US-lessac-medium.onnx \
  --ambient-wav models/audio/hospital-ambience.wav \
  --ambient-volume 0.04 \
  --ambient-ducking 0.35 \
  --model-size tiny \
  --model-dir models
```

If very short or synthetic audio gets filtered out, disable VAD:

```bash
python -m qloopbb.app --duration 5 --no-vad --no-speak
```

## Run Live

Live mode keeps listening for one utterance, transcribes it, routes it, replies, then starts listening again.

```bash
python -m qloopbb.app --live --model-size tiny --model-dir models
```

Stop it with Ctrl+C.

If the live listener starts too easily or misses your voice, tune the simple energy VAD:

```bash
python -m qloopbb.app --live --vad-threshold 0.01 --silence-seconds 1.0
```

## Run Live With Retrieval

Use `--retrieve` to load the local embedding search layer.
The router decides whether a turn actually needs retrieval.
Retrieval mode automatically asks Whisper for English transcripts because downstream thinking is English-only.
The app still keeps the detected source language as the response language for future output translation and TTS.

```bash
python -m qloopbb.app --live --retrieve --model-size tiny --model-dir models
```

Say things like:

- "I need to schedule an MRI."
- "I have knee pain and need a doctor."
- "I need help with my insurance claim."
- "What are the visiting hours?"

The app will print the transcript, route, tool result when one runs, retrieval matches when RAG is needed, and then reply.
Scheduling and billing requests use deterministic local tool stubs.
Department and policy lookups use retrieval when `--retrieve` is enabled.

## Run With Gemini Replies

Set `GEMINI_API_KEY` in your shell before starting the app:

```bash
export GEMINI_API_KEY="your-key-here"
```

You can also put it in a root `.env` file:

```text
GEMINI_API_KEY=your-key-here
```

The app loads `.env` automatically before validating Gemini settings.

Then enable Gemini after routing, tools, and optional retrieval:

```bash
python -m qloopbb.app --live --llm gemini --retrieve --timings --model-size tiny --model-dir models --no-speak
```

The app still keeps deterministic routing in front of the LLM.
Gemini receives the English transcript, route, local tool result, retrieval context when available, and recent conversation history.
Empty transcripts and medical-advice or emergency handoffs skip Gemini and use deterministic safe replies.

Streaming ASR can use the same reply generator:

```bash
python -m qloopbb.app --streaming-asr --llm gemini --retrieve --timings --model-size tiny --model-dir models --no-speak
```

Gemini reply generation still produces English.
Output localization runs after that and before TTS.

Force Gemini output localization:

```bash
python -m qloopbb.app \
  --streaming-asr \
  --llm gemini \
  --output-localizer gemini \
  --response-language hi \
  --retrieve \
  --timings \
  --model-size tiny \
  --model-dir models \
  --no-speak
```

In `auto` mode, output localization uses Gemini when `--llm gemini` is active.
Use `--output-localizer none` to keep replies in English.

## Run A Full Text Pipeline Smoke Test

Use `--input-text` to bypass the microphone and test the rest of the pipeline:

```bash
python -m qloopbb.app \
  --input-text "I need to schedule an MRI" \
  --llm gemini \
  --output-localizer gemini \
  --response-language hi \
  --retrieve \
  --timings \
  --model-size tiny \
  --model-dir models \
  --no-speak
```

This runs:

```text
English input text -> route -> tool and/or retrieval -> Gemini English reply -> Gemini localization -> TTS handoff
```

With `--no-speak`, the final TTS stage is validated as a handoff but no audio is played.

## Async Thinking Phrases

For routes that need tool or retrieval context, the app now starts context resolution in a background worker and immediately emits a short filler phrase.
That filler is printed as `Thinking phrase:` and is spoken through the configured TTS backend unless `--no-speak` is set.

Examples:

- Appointment route: "Let me check the scheduling details."
- Billing route: "Let me pull up the billing desk details."
- Department lookup route: "Let me check the right department."
- Policy lookup route: "Let me look that up."

Disable this with:

```bash
python -m qloopbb.app --live --retrieve --no-thinking-phrases --model-size tiny --model-dir models
```

The final reply still waits for the tool or retrieval context before generation.
This is an async latency-mask layer, not full token-streaming generation yet.

You can override the future output language explicitly:

```bash
python -m qloopbb.app --live --retrieve --response-language hi --model-size tiny --model-dir models
```

To inspect latency by stage:

```bash
python -m qloopbb.app --live --retrieve --timings --model-size tiny --model-dir models --no-speak
```

The timing output separates model load, listening, Whisper transcription, retrieval, reply generation, and speech output.

## Run Streaming ASR With Retrieval Prewarm

Streaming ASR mode records continuously, runs rolling partial Whisper passes, and prewarms retrieval before the utterance ends.
It still does one final Whisper pass after endpointing before replying.
Partial retrieval prewarm only runs when the partial transcript routes to a retrieval-backed lookup.

```bash
python -m qloopbb.app --streaming-asr --retrieve --timings --model-size tiny --model-dir models --no-speak
```

This is the mode to use when testing whether partial speech can prepare retrieval early.
Watch these timing lines:

- `Timing summary: first partial transcript after speech start`
- `Timing summary: first retrieval prewarm after speech start`
- `Timing summary: streaming warm roundtrip after endpoint`

## Current Routing Behavior

The router is deterministic for now.
It classifies English working transcripts into:

- `empty`
- `direct`
- `emergency`
- `medical_advice`
- `appointment`
- `billing`
- `department_lookup`
- `policy_lookup`

Emergency and medical-advice routes return the receptionist-safe handoff.
Appointment, billing, and department routes can call local stub tools.
Department and policy lookup routes can use retrieval when the embedding layer is enabled.
The reply generator can be either the deterministic fixed generator or Gemini.
The output localizer can be disabled, set to Gemini explicitly, or left on `auto`.

## Run With Indian-Language Speech

Use Whisper transcription when you want the original-language transcript:

```bash
python -m qloopbb.app --duration 6 --language hi
```

Use Whisper translation when you want non-English speech translated to English:

```bash
python -m qloopbb.app --duration 6 --language hi --translate
```

## Run With A WAV File

```bash
python -m qloopbb.app --input-wav path/to/input.wav --no-speak
```

## Run Local Embeddings

The embedding slice uses `fastembed` with `BAAI/bge-large-en-v1.5` by default.
This is a stronger English retrieval model and downloads about 1.2 GB on first use.
The architecture expects non-English speech to be translated to English before embedding or retrieval.

```bash
python -m qloopbb.embed --query "I need to schedule an MRI"
```

Add `--timings` to measure embedding model load, index build, and search:

```bash
python -m qloopbb.embed --query "I need to schedule an MRI" --timings
```

You can pass your own documents:

```bash
python -m qloopbb.embed \
  --query "insurance claim status" \
  --doc "Billing helps with insurance claims and invoices." \
  --doc "Radiology schedules MRI and CT scan appointments."
```

## Tests

```bash
python -m unittest discover
```
# qloopbb
