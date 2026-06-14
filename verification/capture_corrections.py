# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Capture a frame-by-frame trace of partial -> final -> supersede rendering.

Drives the in-repo scripted streaming engine through a REAL protocol session with
paced feeding (so partials are not coalesced away), and after each event prints
the event plus the reducer's rendered transcript lines. This is the reproducible
evidence that the live UI renders interim partials, promotes them to finals, and
re-renders corrections (``supersede``) in real time.

Run:  uv run python verification/capture_corrections.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make the app importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from standard_asr import AudioFormat, discover_models  # noqa: E402

from standard_asr_live.engine_view import LiveTranscript, SegmentView  # noqa: E402

_KEY = "scripted/demo"


def _line(seg: SegmentView) -> str:
    """Render one segment view as a plain string with stable/unstable markers.

    The frozen prefix is wrapped in ``[...]`` (the part the engine has frozen and
    will not change); the unsettled tail follows in ``<...>``. A final/closed
    segment is shown plain (settled).

    Args:
        seg: The segment view.

    Returns:
        A plain-text rendering of the segment for the trace.
    """
    if seg.state in ("final", "closed"):
        return f"  [{seg.state:6}] {seg.text!r}"
    return f"  [partial] [{seg.stable_text}]<{seg.unstable_text}>"


async def main() -> int:
    """Drive the scripted session with pacing and print a per-event trace.

    Returns:
        Process exit code (0 on success, 2 if the engine is not installed).
    """
    registry = discover_models()
    if _KEY not in registry.names():
        print(f"{_KEY} not installed; run: uv pip install -e tests/scripted_engine")
        return 2
    engine = registry.create(_KEY)
    fmt = AudioFormat(encoding="pcm_s16le", sample_rate=16000, channels=1)
    state = LiveTranscript()

    async def paced_audio():
        # Feed slowly so the session loop interleaves our partial events between
        # chunks instead of coalescing them (real-time-like pacing).
        for _ in range(12):
            yield b"\x00\x00" * 1600  # 100 ms of silence
            await asyncio.sleep(0.12)

    print("=" * 78)
    print("LIVE CORRECTION TRACE — scripted/demo through a real protocol session")
    print("=" * 78)
    async with engine.start_transcription(audio_format=fmt) as session:
        session.feed(paced_audio())
        step = 0
        async for event in session:
            state.apply(event)
            step += 1
            tag = ""
            if event.type == "supersede":
                tag = f"  old={event.old_ids} -> new={event.new_ids}"
            elif event.type == "error":
                tag = f"  code={event.code} recoverable={event.recoverable}"
            print(f"\n[{step:02d}] EVENT {event.type}{tag}")
            for seg in state.live_segments():
                print(_line(seg))
            for retired in state.retired:
                print(f"  [RETIRED] {retired.text!r}  (struck through on screen)")
        result = session.result()
        diagnostics = session.diagnostics()

    print("\n" + "=" * 78)
    print(f"FINAL committed transcript: {result.text!r}")
    print(f"recoverable errors shown  : {[e.code for e in state.recoverable_errors]}")
    print(f"suppression diagnostics   : {[(d.level, d.code) for d in diagnostics]}")
    print(f"counts                    : {dict(state.counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
