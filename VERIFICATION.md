# VERIFICATION

This is a reproducible report of what was actually run and the results. A reader
can re-run every command below from the repo root and reproduce the evidence in
`verification/`.

- **Machine:** Apple M5 Max, arm64, macOS 27 (Metal, no CUDA).
- **Toolchain:** `uv` 0.11.21, `ffmpeg` 8.1.1 on PATH, Python pinned to 3.12.
- **`standard-asr`:** the `refactor/v0.1.0-redesign` branch (installed from the
  local monorepo checkout for offline dev; the published `pyproject.toml` pins
  the public git branch).

## 0. Setup (one time)

```bash
cd standard-asr-live
uv python pin 3.12
uv venv --python 3.12
uv pip install -e .                                   # the app + standard-asr

# Engines (discovered via entry points; the app never imports them):
uv pip install -e ../standard_asr/cookbook/std_dummy_asr            # batch dummy
uv pip install -e ../standard_asr/cookbook/std_faster_whisper       # real ASR (batch)
uv pip install -e tests/scripted_engine                             # streaming test double
uv pip install -e verification/std_faster_whisper_tiny              # faster-whisper/tiny preset

# Dev tools (tests + lint):
uv pip install "pytest>=8.4.1" "pytest-cov>=7" "pytest-asyncio>=0.24" "ruff>=0.6"
```

> The cookbook ships only `faster-whisper/large-v3` / `distil-large-v3` /
> `turbo`. `verification/std_faster_whisper_tiny` is a 30-line engine-author-style
> preset adding `faster-whisper/tiny` so the real-audio demo runs fast on CPU. It
> is a *plugin*, not part of the app, discovered the same way every engine is.

## 1. Discovery — `models` (zero-config)

```bash
uv run standard-asr-live models
```

Result (the app discovered every installed engine via entry points and read each
one's streaming capabilities **without instantiating** it):

```
                        Discovered Standard ASR engines
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ model key                      ┃ streaming  ┃ description                    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ dummy/echo                     │ batch only │ Dummy echo engine …            │
│ faster-whisper/large-v3        │ batch only │ Standard ASR wrapper …         │
│ faster-whisper/tiny            │ batch only │ faster-whisper tiny …          │
│ scripted/demo                  │ mic, partials, corrections,   │ Scripted … │
│                                │ stable-prefix, closed         │            │
└────────────────────────────────┴────────────┴────────────────────────────────┘
```

`uv run standard-asr-live show scripted/demo` prints full properties, the
capability tree (canonical JSON), and the init-config schema — see
`verification/show_scripted.txt`.

## 2. Real audio through a real engine (faster-whisper `tiny`, batch)

The test file is 48 kHz stereo AAC, ≈57 s English. The engine is batch-only, so
the app **detects batch mode** and runs `transcribe` with a spinner, then renders
the result through the same reducer/view as streaming and exports subtitles.

```bash
STANDARD_ASR_ALLOW_DOWNLOAD=1 uv run standard-asr-live run faster-whisper/tiny \
  --file ../standard_asr/reference/standard_asr_test_audio_english.m4a \
  --language en --plain --no-prompt \
  --export verification/real_audio_out \
  --json-events verification/real_audio_events.jsonl
```

**Actual transcript produced** (7 segments, detected language `en`, duration
57.45 s):

> This is a crazy interesting test for testing the capabilities and initial
> prototype of standard ASR package. By doing this we are creating a sample
> plugin implementation for faster whisper and Q and 3AASR. Both are really good
> ASR engine, so we're going to try out and see if we can implement the plugin
> for these two ASR engines. By doing this we will be able to understand the
> potential issues and whether our design actually working in the real real world
> scenario, because we have been nearly designing things for a very long time.
> Now is the time to put the design into test, complete.

Exports written to `verification/real_audio_out/` (`transcript.txt` / `.srt` /
`.vtt`). The SRT carries indexed, timestamped cues from the engine's segments,
e.g.:

```
1
00:00:00,000 --> 00:00:07,800
This is a crazy interesting test for testing the capabilities and initial prototype of
```

This proves: **engine-agnostic discovery + real transcription + batch fallback +
export**, all through the protocol. The *same app* runs the streaming engine in §3.

## 3. Live corrections — `partial → final → supersede` (the centerpiece)

Neither cookbook engine implements streaming, so corrections are proven against
the in-repo **scripted streaming engine** (`tests/scripted_engine`), a fully
compliant `StandardASR` engine declaring `streaming_input` +
`streaming.emits_partials` + `streaming.re_segments` + `streaming.word_stability`.
It is driven through the **real** `discover_models → start_transcription → feed →
events` path — nothing is faked at the protocol layer.

```bash
# Frame-by-frame trace (paced so interim partials are not coalesced away):
STD_SCRIPTED_STEP_DELAY=0.15 uv run python verification/capture_corrections.py
```

**Actual trace** (`verification/correction_trace.txt`; `[frozen]<unsettled>`
shows the `stable_until` boundary the UI renders):

```
[01] EVENT partial
  [partial] []<the quik>                         ← interim guess (note the typo)
[02] EVENT partial
  [partial] [the ]<quick brown>                  ← frozen prefix grew; typo fixed
[03] EVENT final
  [final ] 'the quick brown fox'                 ← segment settles
[04] EVENT supersede  old=['seg-0'] -> new=['seg-1', 'seg-2']
  [RETIRED] 'the quick brown fox'  (struck through on screen)   ← CORRECTION
[05] EVENT partial
  [partial] [the ]<quick>                         ← replacement streams in live
[06] EVENT final
  [final ] 'the quick '
[07] EVENT partial
  [final ] 'the quick '
  [partial] [brown ]<fox>                         ← second replacement, live
[08] EVENT final
  [final ] 'the quick '
  [final ] 'brown fox jumps'
[09] EVENT partial
  [partial] [over the]< lazy dog>
[10] EVENT final
  [final ] 'the quick '
  [final ] 'brown fox jumps'
  [closed] 'Over the lazy dog.'                   ← closed REWRITES (caps + period)
[11] EVENT error  code=content_lost recoverable=True   ← banner; stream continues
...
FINAL committed transcript: 'the quick  brown fox jumps Over the lazy dog.'
recoverable errors shown  : ['content_lost']
suppression diagnostics   : []                    ← script is fully spec-compliant
counts                    : {'partial': 5, 'final': 4, 'supersede': 1, ...}
```

This proves every must-have: **interim partials render, promote to finals, and a
`supersede` re-renders the correction live** (seg-0 removed, seg-1+seg-2 stream
in), the **frozen prefix** (`stable_until`) is drawn, a **`closed`** final
**replaces** (not appends) post-processed text, and a **recoverable error** is
survived.

Run it as the full app too (final rendered frame + export + JSONL event log):

```bash
uv run standard-asr-live run scripted/demo --file verification/scripted_silence.wav \
  --no-pace --plain --no-prompt \
  --export verification/scripted_out --json-events verification/scripted_events.jsonl
```

The exported `verification/scripted_out/transcript.srt` contains only the
**post-correction** segments (the retired `seg-0` is gone), and
`verification/scripted_events.jsonl` contains the `supersede` event:

```json
{"type": "supersede", "finality": "final", "audio_processed_until": 2.0, "old_ids": ["seg-0"], "new_ids": ["seg-1", "seg-2"], "extra": {}}
```

## 4. Sync bridge (`SyncSession`) parity

The same scripted session driven through the protocol's synchronous bridge yields
the identical corrected transcript:

```bash
uv run standard-asr-live run scripted/demo --file verification/scripted_silence.wav \
  --plain --no-prompt --sync
# -> FINAL RESULT transcript: the quick  brown fox jumps Over the lazy dog.
```

## 5. Fail-loud behaviours (explicit > implicit)

```bash
# A mic request against a batch-only engine fails loudly (does not silently
# fall back), per the protocol philosophy:
uv run standard-asr-live run dummy/echo --mic --no-prompt --plain
# -> error: Engine 'dummy' does not declare 'streaming_input', so it cannot
#    accept live microphone audio. Use --file ...
```

## 6. Tests

```bash
uv run pytest
```

Result: **60 passed**, 77% line coverage overall — the event→view reducer
(`engine_view.py`) is at **97%** and the renderer (`view.py`) at 95%. The reducer
tests (`tests/test_engine_view.py`, 17 cases) feed scripted event lists and
assert state exactly, including every event type, merge/split `supersede`, the
frozen-prefix split, and a cross-check against the protocol's own
`reduce_event`. `tests/test_streaming_e2e.py` drives the scripted engine through
real async **and** `SyncSession` sessions.

```bash
uv run ruff check src tests/scripted_engine verification    # -> All checks passed!
```

## Evidence files (in `verification/`)

| File | What it shows |
|---|---|
| `correction_trace.txt` | Frame-by-frame partial→final→supersede→closed trace |
| `real_audio_out/transcript.{txt,srt,vtt}` | Real faster-whisper transcript + subtitles |
| `real_audio_events.jsonl` | Event log of the real-audio run |
| `scripted_out/transcript.{txt,srt,vtt}` | Post-correction export from the scripted engine |
| `scripted_events.jsonl` | Event log incl. the `supersede` event |
| `show_scripted.txt` | `show` output: properties + capability tree + config schema |
| `capture_corrections.py` | The reproducible trace script |
| `std_faster_whisper_tiny/` | The `faster-whisper/tiny` preset plugin used above |
