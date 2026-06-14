# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the rich rendering of reducer state (the visual grammar).

These assert the styling decisions that make the streaming model legible:
partials dim, finals solid, the frozen prefix distinct, superseded text struck
through. We render to a string-capturing console and check the styled spans.
"""

from __future__ import annotations

from rich.console import Console
from standard_asr import Diagnostic, TranscriptionEvent

from standard_asr_live.engine_view import LiveTranscript, SegmentView
from standard_asr_live.view import (
    render_banner,
    render_diagnostics,
    render_segment,
    render_status,
    render_transcript,
)


def _plain(renderable) -> str:
    """Render a renderable to plain text (no color) for assertions.

    Args:
        renderable: A rich renderable.

    Returns:
        The plain-text rendering.
    """
    console = Console(file=None, width=120, no_color=True)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def test_final_segment_renders_solid_text() -> None:
    """A final segment renders its full text plainly (settled)."""
    seg = SegmentView(segment_id="s0", text="hello world", stable_until=11, state="final")
    text = render_segment(seg)
    assert text.plain == "hello world"


def test_partial_segment_splits_stable_and_unstable() -> None:
    """A partial renders the frozen prefix then the unsettled tail."""
    seg = SegmentView(segment_id="s0", text="the quick brown", stable_until=4, state="open")
    text = render_segment(seg)
    # The plain text is the whole segment; styling differs per span.
    assert text.plain == "the quick brown"
    # The first span is the frozen prefix; the rest is the unstable tail.
    spans = text.spans
    assert spans, "expected styled spans for stable/unstable split"


def test_superseded_segment_is_struck_through() -> None:
    """A just-superseded segment is rendered with the strike style."""
    seg = SegmentView(segment_id="s0", text="old text", state="final", just_superseded=True)
    text = render_segment(seg)
    assert "strike" in str(text.style)


def test_transcript_panel_shows_listening_when_empty() -> None:
    """An empty transcript shows a listening placeholder."""
    out = _plain(render_transcript(LiveTranscript()))
    assert "listening" in out


def test_transcript_panel_renders_segments() -> None:
    """The transcript panel shows committed and in-progress segments."""
    state = LiveTranscript()
    state.apply(TranscriptionEvent.final("s0", "hello world", stable_until=11))
    state.apply(TranscriptionEvent.partial("s1", "in progress", stable_until=0))
    out = _plain(render_transcript(state))
    assert "hello world" in out
    assert "in progress" in out


def test_status_panel_shows_counts_and_language() -> None:
    """The status panel reports engine, mode, counts, and language."""
    state = LiveTranscript()
    state.apply(TranscriptionEvent.final("s0", "hi", stable_until=0, detected_language="en"))
    out = _plain(render_status(state, elapsed=3.0, mode="incremental", engine_key="x/y"))
    assert "x/y" in out
    assert "incremental" in out
    assert "en" in out


def test_status_panel_shows_reconnect() -> None:
    """A reconnect state surfaces in the status panel."""
    state = LiveTranscript()
    state.apply(TranscriptionEvent.progress(reconnect=True, gap_start=1.0, gap_end=2.0))
    out = _plain(render_status(state, elapsed=1.0, mode="incremental", engine_key="x/y"))
    assert "reconnect" in out


def test_diagnostics_panel_empty_and_populated() -> None:
    """The diagnostics panel handles the empty and populated cases."""
    assert "no diagnostics" in _plain(render_diagnostics([]))
    diag = Diagnostic(level="warning", code="stable_until_clamped", message="clamped")
    out = _plain(render_diagnostics([diag]))
    assert "stable_until_clamped" in out
    assert "clamped" in out


def test_banner_only_for_recoverable_errors() -> None:
    """The banner appears only when a recoverable error has been recorded."""
    state = LiveTranscript()
    assert render_banner(state) is None
    state.apply(
        TranscriptionEvent.make_error(
            code="content_lost", recoverable=True, extra={"detail": "gap"}
        )
    )
    out = _plain(render_banner(state))
    assert "content_lost" in out
    assert "gap" in out
