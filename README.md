# standard-asr-live

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
- **Mic or file input.** Live microphone capture (`--mic`) or stream an audio
  file in real-time-like chunks (`--file`) — the file path exercises the full
  live UI without a physical mic.
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
uv python pin 3.12
uv venv --python 3.12
uv pip install -e .                                   # app + standard-asr
```

Then install at least one ASR engine plugin (the app discovers it automatically):

```bash
# A real engine (faster-whisper, CPU-friendly with the tiny model):
uv pip install -e ../standard_asr/cookbook/std_faster_whisper
uv pip install -e verification/std_faster_whisper_tiny     # adds faster-whisper/tiny

# A streaming engine to see live corrections (a protocol test double):
uv pip install -e tests/scripted_engine                    # adds scripted/demo
```

> The published manifest pins `standard-asr` to its public git branch:
> `uv add "standard-asr @ git+https://github.com/standard-voice/standard_asr.git@refactor/v0.1.0-redesign"`.
> For local development this repo resolves it from the sibling monorepo checkout
> via `[tool.uv.sources]`.

## Usage

```bash
# List installed engines and their streaming capabilities
uv run standard-asr-live models

# Inspect one engine in full
uv run standard-asr-live show faster-whisper/tiny

# Live transcription from a FILE (default demo audio if --file omitted)
STANDARD_ASR_ALLOW_DOWNLOAD=1 uv run standard-asr-live run faster-whisper/tiny \
  --file path/to/audio.m4a --language en --export ./out

# See partial -> final -> supersede corrections render LIVE (streaming engine)
STD_SCRIPTED_STEP_DELAY=0.15 uv run standard-asr-live run scripted/demo \
  --file verification/scripted_silence.wav

# Live transcription from the MICROPHONE (needs a streaming_input engine + mic)
uv run standard-asr-live run scripted/demo --mic
uv run standard-asr-live run --list-devices          # discover input devices

# Drive via the synchronous SyncSession bridge instead of async
uv run standard-asr-live run scripted/demo --file audio.wav --sync

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
`uv run standard-asr-live run <streaming-engine> --mic` (use `--list-devices` /
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
uv run pytest          # 62 tests; the reducer is unit-tested with scripted streams
uv run ruff check src tests
```

The most important tests (`tests/test_engine_view.py`) feed scripted event lists
into the reducer and assert state exactly — including every event type, merge/split
`supersede`, the frozen-prefix split, and a cross-check against the protocol's own
`reduce_event`. `tests/test_streaming_e2e.py` drives the in-repo scripted
streaming engine through real async **and** `SyncSession` sessions.

## License

Apache-2.0.
