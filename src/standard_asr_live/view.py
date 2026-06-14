# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Rich rendering of the live transcript, status, and diagnostics.

Pure presentation: it reads a :class:`~standard_asr_live.engine_view.LiveTranscript`
(the reducer state) and paints it. It never touches the engine or audio. The
visual grammar mirrors the streaming model:

* **final / closed** segment text -> solid (settled).
* **partial** segment -> frozen prefix (``text[:stable_until]``) solid; unsettled
  tail (``text[stable_until:]``) dim + italic. This draws ``stable_until`` -- the
  frozen-prefix frontier -- literally on screen.
* a just-**superseded** region -> struck through / dimmed for one frame, so a
  correction is visibly replaced rather than silently swapped.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .engine_view import LiveTranscript, SegmentView

if TYPE_CHECKING:
    from standard_asr import Diagnostic

#: Style for settled (final / closed) text.
_STYLE_FINAL = "bold white"
#: Style for the frozen prefix of an in-progress partial (settled-but-open).
_STYLE_STABLE = "cyan"
#: Style for the unsettled tail of a partial (still changing).
_STYLE_PARTIAL = "dim italic"
#: Style for a segment being retired by a supersede (one-frame highlight).
_STYLE_SUPERSEDED = "strike dim red"


def _fmt_time(seconds: float) -> str:
    """Format seconds as ``MM:SS.s``.

    Args:
        seconds: A non-negative duration in seconds.

    Returns:
        A compact ``MM:SS.s`` string.
    """
    td = timedelta(seconds=max(0.0, seconds))
    total = td.total_seconds()
    minutes, secs = divmod(total, 60)
    return f"{int(minutes):02d}:{secs:04.1f}"


def render_segment(seg: SegmentView) -> Text:
    """Render one segment view with partial/stable/final styling.

    Args:
        seg: The segment view to render.

    Returns:
        A styled :class:`rich.text.Text` line for the segment.
    """
    if seg.just_superseded:
        return Text(seg.text or " ", style=_STYLE_SUPERSEDED)
    if seg.state in ("final", "closed"):
        return Text(seg.text, style=_STYLE_FINAL)
    # Open (partial): solid frozen prefix + dim italic unsettled tail.
    line = Text()
    if seg.stable_text:
        line.append(seg.stable_text, style=_STYLE_STABLE)
    line.append(seg.unstable_text, style=_STYLE_PARTIAL)
    return line


def render_transcript(state: LiveTranscript) -> RenderableType:
    """Render the transcript panel from the reducer state.

    Args:
        state: The live transcript reducer state.

    Returns:
        A renderable transcript panel.
    """
    lines: list[Text] = []
    # Show segments retired this frame (highlighted) interleaved by position is
    # complex; instead show them as a trailing "corrected" strip for one frame.
    for seg in state.live_segments():
        rendered = render_segment(seg)
        if rendered.plain.strip():
            lines.append(rendered)
    for retired in state.retired:
        lines.append(render_segment(retired))
    if not lines:
        lines = [Text("(listening...)", style="dim")]
    body = Group(*lines)
    title = "TRANSCRIPT"
    return Panel(body, title=title, title_align="left", border_style="blue", padding=(0, 1))


def render_status(
    state: LiveTranscript, *, elapsed: float, mode: str, engine_key: str
) -> RenderableType:
    """Render the status panel (counters, cursor, language).

    Args:
        state: The live transcript reducer state.
        elapsed: Wall-clock seconds since the session started.
        mode: The transcription mode label (incremental / whole_input / batch).
        engine_key: The model key being run.

    Returns:
        A renderable status panel.
    """
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", style="dim")
    table.add_column()
    table.add_row("engine", engine_key)
    table.add_row("mode", mode)
    table.add_row("elapsed", _fmt_time(elapsed))
    table.add_row("audio", _fmt_time(state.audio_processed_until))
    counts = state.counts
    table.add_row(
        "events",
        f"{sum(counts.values())}  "
        f"(partial {counts.get('partial', 0)}, final {counts.get('final', 0)}, "
        f"supersede {counts.get('supersede', 0)}, progress {counts.get('progress', 0)})",
    )
    committed = sum(1 for s in state.live_segments() if s.state in ("final", "closed"))
    table.add_row("segments", f"{committed} committed / {len(state.order)} shown")
    table.add_row("language", state.detected_language or "—")
    if state.reconnecting:
        gap = state.last_gap or (None, None)
        table.add_row("[yellow]reconnect", f"[yellow]gap {gap[0]}–{gap[1]}")
    return Panel(table, title="STATUS", title_align="left", border_style="green", padding=(0, 1))


def render_diagnostics(diagnostics: list[Diagnostic]) -> RenderableType:
    """Render the diagnostics panel.

    Args:
        diagnostics: The session/result diagnostics to display.

    Returns:
        A renderable diagnostics panel (a friendly note when there are none).
    """
    if not diagnostics:
        body: RenderableType = Text("no diagnostics", style="dim")
    else:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="dim", justify="left")
        table.add_column()
        for diag in diagnostics[-12:]:  # keep the panel bounded on screen
            level_style = "yellow" if diag.level == "warning" else "cyan"
            table.add_row(Text(diag.level, style=level_style), Text(diag.code))
            if diag.message:
                table.add_row("", Text(diag.message, style="dim"))
        body = table
    return Panel(
        body, title="DIAGNOSTICS", title_align="left", border_style="magenta", padding=(0, 1)
    )


def render_banner(state: LiveTranscript) -> RenderableType | None:
    """Render a warning banner for recoverable errors / reconnects, if any.

    Args:
        state: The live transcript reducer state.

    Returns:
        A renderable banner, or ``None`` when there is nothing to warn about.
    """
    if not state.recoverable_errors:
        return None
    last = state.recoverable_errors[-1]
    detail = last.extra.get("detail", "") if last.extra else ""
    msg = Text()
    msg.append("recoverable: ", style="bold yellow")
    msg.append(last.code or "error", style="yellow")
    if detail:
        msg.append(f" — {detail}", style="dim")
    return Panel(msg, border_style="yellow", padding=(0, 1))


__all__ = [
    "render_banner",
    "render_diagnostics",
    "render_segment",
    "render_status",
    "render_transcript",
]
