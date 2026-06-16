# standard-asr-live

> ⚠️ **Experimental — for protocol testing.** This is an experimental demo app for the [Standard ASR](https://github.com/standard-voice/standard_asr) protocol, published to exercise the interface end-to-end. Expect breaking changes; it is not production-ready. It may later be folded into `standard-asr` itself.

**Live streaming speech-to-text in your terminal, with real-time on-screen
corrections — against _any_ installed [Standard ASR](https://github.com/standard-voice/standard_asr) engine.**

`standard-asr-live` is a polished reference application for the Standard ASR
protocol. Point it at a microphone or an audio file, pick any compliant ASR
plugin you have installed, and watch the transcript appear live — interim guesses
rendered dim, settling into solid finals, with **re-segmentation corrections
(`supersede`) re-rendering in real time** and the engine's frozen-prefix
(`stable_until`) drawn right on screen.

It depends **only** on the protocol (`standard-asr`) plus UI/audio libraries. It
never imports a concrete engine — every engine is discovered via entry points.
This app *is* the proof of the protocol's headline promise: **write once, run with
any engine**, streaming semantics included.

```
┌─ standard-asr-live ─ scripted/demo ─ incremental ────────────────────────────┐
│  TRANSCRIPT                                                                   │
│  the quick                          ← final (solid)                          │
│  brown fox jumps                    ← final (solid)                          │
│  Over the lazy[ dog]<...>           ← stable prefix solid, unsettled tail dim│
├─ STATUS ──────────────────────────────┬─ DIAGNOSTICS ────────────────────────┤
│ events 11 (partial 5, final 4, sup 1) │ no diagnostics                       │
│ audio 00:04.0   language en           │                                      │
└────────────────────────────────────────┴──────────────────────────────────────┘
```

---

## Why this exists

Real-time ASR is the most fragmented part of the speech ecosystem: some engines
emit interim text that gets rewritten, some never revise, some re-segment after a
second pass. Standard ASR unifies all of that into one event protocol
(`partial` / `final` / `supersede` / `progress` / `done` / `error`) with an
explicit segment lifecycle and a frozen-prefix stability guarantee. This app is a
correct, copy-pasteable implementation of the consumer side — the **event → view
reducer** (`src/standard_asr_live/engine_view.py`) is the heart, and it's pure,
tiny, and unit-tested.

## Features

- **Zero-config engine discovery.** `standard-asr-live models` lists every
  installed plugin and summarizes its streaming capabilities — no instantiation.
- **`show` any engine.** Full properties, the capability tree (canonical JSON),
  and the init-config schema.
- **Settings from the schema.** The config form is generated from the engine's
  JSON Schema; **secret fields are prompted with no echo and never logged**.
- **Mic-first, zero-config.** With no audio file the **microphone is the
  default**, so `standard-asr-live <model>` just works; pass a file path
  (positional, or `--file`) to stream a file in real-time-like chunks instead —
  exercising the full live UI without a physical mic. Ctrl-C stops mic capture
  and finalizes the transcript.
- **The live transcript view.** `partial` text is dim/italic; it promotes to a
  solid `final`; `supersede` removes the retired segments and re-renders the
  replacements live; the `stable_until` frozen prefix is drawn distinctly; a
  diagnostics panel streams the session's diagnostics; progress/elapsed/throughput
  update continuously.
- **Three modes, one app, chosen from capabilities.** Incremental streaming
  (`streaming_input`), whole-input streaming (`streaming_output`), or a **batch
  fallback** with a spinner — the app picks the right one automatically and fails
  loudly when you ask for something an engine can't do (e.g. mic on a batch-only
  engine).
- **Export.** TXT / SRT / VTT on completion (via the protocol's `to_srt`/`to_vtt`).
- **Async or sync.** `--sync` drives the protocol's `SyncSession` bridge instead
  of the async iterator — same UI, both consumer styles proven.
- **Honest by default.** Diagnostics surfaced, recoverable errors banner-and-
  continue, terminal errors shown — never a silent wrong transcript.

UI: [`rich`](https://github.com/Textualize/rich). Audio:
[`sounddevice`](https://python-sounddevice.readthedocs.io) (mic) + `ffmpeg`/
[`soundfile`](https://python-soundfile.readthedocs.io) (file decode).

## Install

Requires Python 3.12 (pinned), `ffmpeg` on PATH, and `uv`.

```bash
cd standard-asr-live
uv sync                       # the app + the protocol + a built-in scripted/demo engine
```

`uv sync` installs `standard-asr-live`, the `standard-asr` protocol, and a
`scripted/demo` streaming test engine — so you can watch live corrections
immediately, **with no model download and no real engine installed**. The app
depends on **no** ASR engine; `[tool.uv.sources]` only resolves the protocol
itself from the co-located monorepo checkout during development.

### Add a real ASR engine — _any_ compliant plugin

The app never imports or bundles an engine. Install **any** Standard ASR-compliant
plugin into the same environment and it appears automatically via entry-point
discovery — no app config, no code change, no per-engine integration:

```bash
uv pip install git+https://github.com/standard-voice/std-mlx-audio       # Apple-Silicon (MLX): Qwen3-ASR, Whisper, Parakeet, +more
uv pip install git+https://github.com/standard-voice/std-faster-whisper  # faster-whisper (CPU-friendly)
uv pip install git+https://github.com/standard-voice/std-qwen3-asr       # Qwen3-ASR via a client backend
# ...or ANY other compliant plugin — including one you wrote yourself.
```

(Once these publish to PyPI, it's simply `uv pip install std-mlx-audio`.) Then
list what's now discoverable — switching engines is a one-line model-key change,
never an app change:

```bash
uv run standard-asr-live models
```

This *is* the protocol's headline promise in action: **the application does not
know, and does not need to know, which ASR engine it is talking to.**

## Usage

```bash
# Just run it: pick a model interactively, transcribe from the microphone
uv run standard-asr-live

# A specific model, from the microphone (`run` is the default command)
uv run standard-asr-live scripted/demo

# Transcribe a FILE with any installed model (run `models` to get a <model-key>)
STANDARD_ASR_ALLOW_DOWNLOAD=1 uv run standard-asr-live <model-key> \
  path/to/audio.m4a --language en --export ./out

# See partial -> final -> supersede corrections render LIVE (streaming engine)
STD_SCRIPTED_STEP_DELAY=0.15 uv run standard-asr-live scripted/demo \
  verification/scripted_silence.wav

# List installed engines, or inspect one in full
uv run standard-asr-live models
uv run standard-asr-live show <model-key>

# Microphone helpers
uv run standard-asr-live --list-devices              # discover input devices
uv run standard-asr-live scripted/demo --mic         # force mic (it is the default)

# Drive via the synchronous SyncSession bridge instead of async
uv run standard-asr-live scripted/demo audio.wav --sync

# Environment / dependency diagnostics
uv run standard-asr-live doctor
```

Useful `run` flags: `--chunk-ms` (wire chunk size), `--speed` / `--no-pace`
(file pacing), `--set KEY=VALUE` (config override, repeatable; secrets prompted),
`--strict-lifecycle` (ask the session to raise on illegal transitions),
`--json-events PATH` (append every event as JSONL), `--plain` (no live redraw, for
non-TTY / CI capture).

### Microphone mode

Mic capture needs OS permission and an interactive terminal, so automated
verification uses the file path (streamed in real-time-like chunks) to drive the
full live UI. Mic mode is fully implemented: run
`uv run standard-asr-live <streaming-engine> --mic` (use `--list-devices` /
`--device N` to choose an input). The app **fails loudly** if the chosen engine
does not declare `streaming_input`.

## How it works (architecture)

```
 audio (file/mic) ─▶ audio_io ──PCM chunks──▶ standard_asr session ──events──▶
                                                                              │
                                          ┌───────────────────────────────────┘
                                          ▼
                              engine_view.LiveTranscript   ← THE REDUCER (pure)
                                  .apply(event) → state
                                          │
                                          ▼
                                   view.py (rich Live)
```

- `engine_view.py` — **the event → view reducer.** Pure, sync, no I/O. Implements
  the spec's canonical `partial`/`final`/`supersede` reduce plus the view state a
  live UI needs (frozen-prefix boundary, counts, reconnect/error banners). This is
  the file to copy into your own app.
- `driver.py` — selects incremental / whole-input / batch **from capabilities**
  and pumps events into the reducer (async and `SyncSession` paths).
- `audio_io.py` — file decode (ffmpeg → `pcm_s16le`) with real-time pacing, and
  mic capture (`sounddevice`).
- `config_form.py` — JSON Schema → settings prompt; secret-safe.
- `view.py` — `rich` rendering of the reducer state.
- `export.py` — TXT/SRT/VTT via the protocol's renderers.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full design and
[`VERIFICATION.md`](VERIFICATION.md) for a reproducible run report (real
transcript, live-correction trace, exports). Protocol DX feedback collected while
building this is in [`docs/STANDARD_ASR_FINDINGS.md`](docs/STANDARD_ASR_FINDINGS.md).

## Tests

```bash
uv run pytest          # the reducer is unit-tested with scripted streams
uv run ruff check src tests
```

The most important tests (`tests/test_engine_view.py`) feed scripted event lists
into the reducer and assert state exactly — including every event type, merge/split
`supersede`, the frozen-prefix split, and a cross-check against the protocol's own
`reduce_event`. `tests/test_streaming_e2e.py` drives the in-repo scripted
streaming engine through real async **and** `SyncSession` sessions.

## License

Apache-2.0.
