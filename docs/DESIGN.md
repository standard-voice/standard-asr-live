# standard-asr-live — Design

> A polished terminal app for **live streaming speech-to-text with real-time
> on-screen corrections**, against *any* installed Standard ASR engine. It is a
> reference application that proves the protocol's headline promise — **"write
> once, run with any engine"** — including unified streaming semantics
> (`partial` / `final` / `supersede` / `progress` / `done` / `error`).

This document is written **before** the implementation and drives it. The heart
of the app is the **event → view reducer** (§4); everything else is plumbing
around it.

---

## 1. Target users

1. **Someone evaluating Standard ASR.** Wants to *see* the protocol work end to
   end: discover plugins with zero config, pick a model, watch live transcription
   with corrections rendering in real time, inspect diagnostics, export subtitles.
   The app is a working demo and a credibility check on the spec.
2. **A developer building a captioning / voice-agent / live-notes app.** Wants a
   *reference implementation* of the streaming event model done correctly —
   particularly the `supersede` reduce and the `stable_until` frozen-prefix
   handling that the spec calls the must-have. They can lift the reducer
   (`engine_view.py`) almost verbatim.
3. **An end user who wants live transcription in the terminal.** Installs a
   plugin (`uv pip install std-faster-whisper`), runs one command, gets live
   captions from a file or microphone, and exports a transcript / SRT / VTT.

The app optimizes for **(1)** and **(2)** without compromising **(3)**.

---

## 2. Engine-agnosticism (hard constraint)

The app depends **only** on `standard-asr` (the protocol) plus UI/audio
libraries. It **never** imports a concrete engine package. Every engine
interaction goes through the protocol:

| Need | Protocol API used |
|---|---|
| List installed engines | `discover_models()` → `ModelRegistry.names()` / `.by_engine()` |
| Engine identity / model key | `ModelRegistry.spec(name)` → `ModelSpec` |
| Properties & capabilities (no instantiation) | `registry.engine_class(name).properties` / `.declared_capabilities` |
| Render a settings form | `registry.config_schema(name)` → JSON Schema (secret fields flagged `secret: true`) |
| Build the engine | `registry.create(name, **config)` |
| Per-feature gating | `engine.supports("streaming.re_segments")`, etc. |
| Batch transcription | `engine.transcribe(audio, RuntimeParams(...))` |
| Streaming session | `engine.start_transcription(audio_format=...)`, `session.feed(chunks)`, `async for event in session` |
| Live result so far | `session.result()` |
| Diagnostics | `session.diagnostics()` and `result.diagnostics` |
| Export | `to_srt(result)`, `to_vtt(result)`, `result.text` |

If the app would ever need to know "which engine is this," that is a design
smell — it must instead ask a capability or property.

---

## 3. Feature set & UX

### 3.1 Command surface

A single CLI, `standard-asr-live`, with subcommands (argparse, stdlib — keep deps
lean, mirror the core CLI's style):

```
standard-asr-live models                 # list discovered engines (+ streaming caps)
standard-asr-live show <model-key>       # full properties + capability tree + config schema
standard-asr-live run  <model-key> [opts]   # the live transcription app (the centerpiece)
standard-asr-live doctor                 # environment / dependency sanity (delegates to standard_asr.diagnose)
```

`run` options (all optional; interactive prompts fill the gaps):

```
--file PATH            stream a file in real-time-like chunks (default demo path if omitted & no --mic)
--mic                  capture from the default microphone (live)
--list-devices         print input devices and exit (mic discovery)
--device N             input device index for --mic
--language TAG         BCP-47 or "auto" (per-request; gated by the engine)
--chunk-ms N           wire chunk size in ms (default 100)
--speed FLOAT          file pacing multiplier (1.0 = real time; higher = faster than real time)
--no-pace              feed the file as fast as possible (stress / CI mode)
--set KEY=VALUE        config field override (repeatable); secret fields prompted, never echoed
--export DIR           after completion, write transcript.txt / .srt / .vtt to DIR
--sync                 drive via the SyncSession bridge instead of async (exercises both code paths)
--strict-lifecycle     ask the session to RAISE on illegal transitions (default: suppress+diagnose)
--json-events PATH     append every event as JSONL (evidence / debugging)
--no-color / --plain   degrade gracefully for non-TTY / CI capture
```

### 3.2 Flow

1. **Discover.** `discover_models()`. If no plugins: explain how to install one
   (point at the cookbook). List models with a one-line streaming-capability
   summary so the user can pick wisely.
2. **Select model.** From CLI arg or an interactive picker. Resolve the
   `ModelSpec`; read `engine_class().properties` and `.declared_capabilities`
   **without instantiating** (so a credentialed engine can still show its form).
3. **Configure.** Read `config_schema(model)`. Render a settings prompt:
   - Skip the `engine` discriminator (entry-point-derived) and policy fields the
     env must not flip (`strict`, `allow_private_urls`) unless `--set` overrides.
   - For each remaining field: show name, type, default, description. Prompt only
     when interactive and the user opts to customize; otherwise accept defaults.
   - **Secret fields** (`json_schema_extra.secret == true`): prompt with
     `getpass` (no echo), never print the value, never write it to logs/JSONL.
   - `--set key=value` overrides win; values are coerced per the schema.
   - Build the engine with `registry.create(model, **config)`.
4. **Choose input.** Mic or file (CLI flag, else prompt). For mic, validate the
   engine can stream incrementally (`supports("streaming_input")`) and capture is
   possible; for file, decode + resample to the session's wire format.
5. **Decide the mode** (the protocol branch):
   - `supports("streaming_input")` → **streaming** path (incremental `feed`).
   - else if `supports("streaming_output")` → **whole-input streaming** path
     (`start_transcription(audio=...)`, file only — mic needs incremental input).
   - else → **batch fallback** (`transcribe`) with a spinner. Mic is rejected
     loudly here (you can't batch an unbounded live source).
   - A mic request against a non-`streaming_input` engine **fails loudly** with a
     clear message (explicit > implicit; never silently degrade).
6. **Live view** (§3.3). Drive the session; the **event → view reducer** (§4)
   folds each event into render state; the renderer paints it.
7. **Complete.** Show the final `TranscriptionResult` (text, detected language,
   duration, segment count), the diagnostics panel, and elapsed/throughput.
   Offer / perform export (TXT / SRT / VTT).

### 3.3 The live transcript view (the centerpiece)

Rendered with **`rich`** (`Live` + `Layout`); chosen over `textual` because the
app is output-dominant (no complex interactive widgets), `rich` has a smaller
dependency surface, and it captures cleanly to text for CI evidence. Layout:

```
┌─ standard-asr-live ─ faster-whisper/large-v3 ─ streaming ─────────────┐
│  TRANSCRIPT                                                           │
│  Hello world this is a test.            ← final (solid)              │
│  ▏the quick brown fox▏ jumps over       ← stable prefix solid,       │
│                                            tail dim/italic (partial) │
│                                                                      │
├─ STATUS ──────────────────────────────┬─ DIAGNOSTICS ───────────────┤
│ elapsed 00:12   audio 00:11.4          │ info  audio_conversion      │
│ events 142  partial 130 final 11 sup 3 │   decoded m4a→pcm, downmix  │
│ rate 1.0x   segments 11                │ warn  stable_until_clamped  │
│ language en (detected)                 │   …                         │
└────────────────────────────────────────┴─────────────────────────────┘
```

Rendering rules (driven by the reducer state, §4):

- **`partial`** segment → its tail (`text[stable_until:]`) shown **dim + italic**
  ("not settled yet"); its frozen prefix (`text[:stable_until]`) shown solid.
- **`final`** segment → whole text **solid** (settled). A `closed` final is shown
  identically but tagged closed in the event log (it may have rewritten text;
  the reducer **replaces**, never appends).
- **`supersede`** → retired `old_ids` segments are **removed** and the new
  `new_ids` segments render as they arrive — visibly, in real time. A brief
  highlight marks the just-superseded region so the correction is *seen*.
- **`stable_until`** is visualized as the solid/dim boundary within a partial — a
  literal picture of the frozen prefix. Engines that report `stable_until=0`
  (`word_stability=false`) simply render the whole partial dim, which is correct.
- **`progress`** → advance the audio cursor / elapsed; if `reconnect`, surface a
  reconnect banner and the gap.
- **`error`** → recoverable errors (e.g. `content_lost`) show as a warning banner
  and the stream continues; a terminal error ends the view with the code shown.
- **`done`** → finalize.
- **Diagnostics panel** streams `session.diagnostics()` live (gating, language
  resolution, lifecycle suppression, clamps) — surfacing the protocol's honesty
  guarantees instead of hiding them.

### 3.4 Graceful degradation (explicit, loud)

- **No streaming support at all** → batch fallback with a spinner; the final
  result still flows through the same result/diagnostics/export path.
- **Engine can't take mic input** (`streaming_input=false`) but `--mic` given →
  **fail loudly** with the exact reason and the fix (use a file, or an engine
  that declares `streaming_input`).
- **No `[audio]` / ffmpeg for file decode** → clear error pointing at the fix.
- **Plugin fails to load / construct** → surface the protocol's error
  (`FactoryLoadError`, `ConfigError`, pydantic `ValidationError`) verbatim, with
  credentials never echoed.

---

## 4. Architecture — the event → view reducer (the heart)

The app is structured so the **pure reducer** is isolated, deterministic, and
unit-testable without any engine, audio, or terminal.

```
              ┌──────────────┐   AudioFormat / RuntimeParams
   audio  ──▶ │  audio_io     │ ─────────────────────────────┐
 (file/mic)   │  (sounddevice │                               ▼
              │   / ffmpeg)   │  Iterable[bytes] (PCM)   ┌──────────────────┐
              └──────────────┘ ───────feed()──────────▶ │  standard_asr     │
                                                         │  session          │
                                                         └────────┬─────────┘
                                                  async for event │ TranscriptionEvent
                                                                  ▼
                                                         ┌──────────────────┐
                                                         │  LiveTranscript   │  ← THE REDUCER
                                                         │  .apply(event)    │     (pure, sync,
                                                         │  → render state   │      no I/O)
                                                         └────────┬─────────┘
                                                                  ▼
                                                         ┌──────────────────┐
                                                         │  rich renderer    │  (LiveView)
                                                         └──────────────────┘
```

### 4.1 `LiveTranscript` — the reducer

A pure state machine. Input: a `TranscriptionEvent`. Output: mutated render
state. **No async, no I/O, no rich** — so tests feed it a scripted event list and
assert the resulting state exactly.

State:

```python
@dataclass
class SegmentView:
    segment_id: str
    text: str
    stable_until: int          # codepoints; text[:stable_until] is frozen
    state: Literal["open", "final", "closed"]   # lifecycle as the app sees it
    superseded: bool = False   # marked for one render then dropped (highlight)
    start: float | None = None
    end: float | None = None

@dataclass
class LiveTranscript:
    order: list[str]                       # segment_ids in arrival/reading order
    segments: dict[str, SegmentView]
    audio_processed_until: float = 0.0
    detected_language: str | None = None
    counts: Counter                        # per event-type tally (status panel)
    reconnecting: bool = False
    last_gap: tuple[float|None, float|None] | None = None
    terminal: TranscriptionEvent | None = None   # the done/error that ended it
    recoverable_errors: list[TranscriptionEvent] # content_lost etc. (banner log)
```

`apply(event)` mirrors the spec's canonical reduce (spec ST §5.2) **exactly**,
extended with the view-only bookkeeping the UI needs:

- `partial`: upsert `segments[id]` with `text` + `stable_until`; state `open`;
  append to `order` if new.
- `final`: same upsert; state `final` (or `closed` if `finality == "closed"`).
  Text is **replaced**, never appended (a `closed` may rewrite/shorten it).
- `supersede`: mark each `old_id` `superseded` (rendered highlighted once) then
  drop it from `order`/`segments`; `new_ids` segments appear as their
  `partial`/`final` events arrive. Disjointness/ordering already guaranteed by
  the protocol's event model, so the reducer trusts the validated event.
- `progress`: set `audio_processed_until`; if `reconnect`, set
  `reconnecting=True` + `last_gap`.
- `error`: if recoverable → append to `recoverable_errors` (banner), keep going;
  if terminal → store `terminal`.
- `done`: store `terminal`.
- Always: bump `counts[type]`; track `detected_language` if present.

**Why mirror the spec reduce rather than only call `session.result()`?** Because
the live view must show *in-progress* `partial` text and the *frozen-prefix*
boundary — neither of which appears in the reduced `TranscriptionResult` (which
contains only committed `final` segments). The app therefore keeps its own view
state for the screen and uses `session.result()` for the authoritative final
output and export. The two are reconciled at `done` (a debug assertion checks the
committed finals match).

### 4.2 Why a reducer, not ad-hoc mutation

- **Determinism / testability.** The single most important thing to test (per the
  brief) is event → view. A pure function makes scripted-stream tests trivial and
  the `supersede` correctness provable.
- **Correctness boundary.** All protocol-shaped logic lives in one ~120-line file.
  The renderer is "dumb" (state → pixels); the session driver is "dumb" (pump
  events into the reducer). Bugs have one home.
- **Reuse.** This is exactly what a real app developer copies. Keeping it pure and
  framework-free maximizes its value as a reference.

### 4.3 Modules

```
src/standard_asr_live/
  __init__.py
  __main__.py          # python -m standard_asr_live
  cli.py               # argparse, subcommands, orchestration, graceful errors
  discovery.py         # thin helpers over discover_models / ModelSpec / capability summaries
  config_form.py       # JSON Schema → settings prompt; secret handling (getpass, never echo)
  engine_view.py       # LiveTranscript + SegmentView  (THE REDUCER — pure)
  audio_io.py          # file → PCM chunks (ffmpeg/soundfile) + mic capture (sounddevice); pacing
  driver.py            # picks streaming / whole-input / batch; pumps events → reducer; async + sync
  view.py              # rich Live rendering of LiveTranscript + status + diagnostics panels
  export.py            # TXT/SRT/VTT writers over to_srt/to_vtt + result.text
  errors.py            # app-level error types + friendly rendering of protocol exceptions
```

### 4.4 Audio path

- **File.** Decode with `ffmpeg` (always present here) to **16-bit signed
  little-endian PCM, mono, at the session's chosen sample rate** (the canonical
  wire encoding `pcm_s16le`). We pick the sample rate by intersecting the
  engine's `accepted_sample_rates` (or `native_sample_rate`, default 16 kHz) with
  what the engine will accept for the wire `AudioFormat`. Chunk into `--chunk-ms`
  frames and feed at real-time pace (`--speed`/`--no-pace` control pacing). This
  exercises the *full* live UI without a physical mic.
- **Mic.** `sounddevice` raw input stream at the chosen sample rate, mono,
  `int16`; push chunks via `session.send_audio` (manual mode) or a queue-backed
  iterable to `feed`. Mic requires `streaming_input`.
- **Wire format.** `AudioFormat(encoding="pcm_s16le", sample_rate=sr, channels=1)`.
  We validate it against the engine up front by letting `start_transcription`
  raise (the standard layer fail-closes on unreachable rate / non-mono).

### 4.5 Async vs sync

The async path is primary (`async for event in session`). `--sync` drives the
**`SyncSession`** bridge instead (`for event in SyncSession(session)`), proving
the protocol's sync mirror works and exercising both consumer styles. The reducer
and renderer are identical for both; only the pump differs.

---

## 5. Demonstrating `supersede` / corrections

Neither cookbook engine (`std-dummy-asr`, `std-faster-whisper`) implements
streaming — both are batch-only (a finding, see FINDINGS). To prove the
correction-rendering UI against the *real protocol types*, the repo ships a tiny
**in-repo compliant streaming engine** used only in tests and an opt-in demo:

- `tests/scripted_engine.py` — a `StandardASR` engine (subclass of `EngineBase`)
  that declares `streaming_input` + `streaming.emits_partials` +
  `streaming.re_segments` + `streaming.word_stability`, and whose session yields a
  **scripted** `partial → partial → final`, then a `supersede` that splits/merges
  segments and re-emits `partial → final`, with growing `stable_until`. This is a
  protocol-level test double (like the cookbook dummy, but for streaming).
- It is registered via a test-only entry point so the *same* discovery →
  `supports` → `start_transcription` path the app uses in production is exercised
  end to end; the app code never references it.

This yields deterministic, reproducible evidence that `partial`/`final`/
`supersede`/`stable_until` all render correctly and that corrections appear live.

For **real audio**, the app runs faster-whisper (cookbook) in **batch** mode
(spinner fallback) — proving the engine-agnostic batch path and producing a
correct English transcript + working SRT/VTT export.

---

## 6. Testing strategy

- **Reducer (most important).** `tests/test_engine_view.py` — feed scripted event
  lists (including every type and the merge/split `supersede` cases and a
  `stable_until` clamp) into `LiveTranscript.apply` and assert state exactly.
  Cross-check against the protocol's own `reduce_event` / `StreamReducer`.
- **End-to-end streaming via the scripted engine.** Discover → create →
  `start_transcription` → drain → assert the final view + `session.result()` +
  diagnostics. Run through **both** async and `SyncSession`.
- **Export.** Render a known result to TXT/SRT/VTT and assert the output.
- **Config form.** Feed a JSON Schema with a secret field and assert the secret
  is never placed in the echoable config dump.
- **CLI smoke.** `models` / `show` against the scripted engine; `run --file` with
  `--no-pace` and `--plain` capturing the final rendered state.

---

## 7. Non-goals / explicit scope limits

- No engine implementation beyond the test scripted double (the repo is an *app*).
- No persistence/server; export is to local files only.
- `stable_until` is treated at the codepoint level per spec (grapheme-cluster
  refinement is a spec SHOULD we surface but don't re-implement).
- Multi-channel / diarized streaming display is out of scope (v1 streaming wire is
  mono); the result panel still shows channels if a batch result carries them.
</invoke>
