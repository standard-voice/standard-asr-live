# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""The event -> view reducer: the heart of standard-asr-live.

:class:`LiveTranscript` folds a stream of :class:`~standard_asr.TranscriptionEvent`
objects into the state the terminal renders. It is the canonical application-side
reduce from the spec (section "Streaming", ST.5.2) -- ``partial`` shows,
``final`` commits, and **``supersede`` removes the retired segments and lets the
replacements render as they arrive** -- extended with the view-only bookkeeping a
live UI needs (a frozen-prefix boundary to draw, per-type counts, a reconnect
banner, a recoverable-error log, the terminal event).

It is deliberately **pure and framework-free**: no async, no I/O, no ``rich``. A
``TranscriptionEvent`` goes in via :meth:`LiveTranscript.apply`; render state
comes out. That makes the single most important behaviour -- correct handling of
every event type, especially ``supersede`` corrections and the monotonic
``stable_until`` frozen prefix -- deterministically unit-testable against a
scripted event list, with no engine, microphone, or terminal in the loop.

Why keep our own view state instead of only calling ``session.result()``? The
reduced :class:`~standard_asr.TranscriptionResult` contains only *committed*
``final`` segments -- it has no in-progress ``partial`` text and no frozen-prefix
boundary. The live screen needs both. So the app keeps this view state for the
display and uses ``session.result()`` for the authoritative final transcript and
export; the two are reconciled at ``done`` (see :meth:`committed_text`).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from standard_asr import TranscriptionEvent

#: Lifecycle state of a segment *as the application sees it* (a projection of the
#: spec's per-segment state machine, ST.5.1, onto what the UI must distinguish).
SegmentState = Literal["open", "final", "closed"]


@dataclass
class SegmentView:
    """Render state for a single segment.

    Args:
        segment_id: The segment's stable id.
        text: The segment's complete current text (cumulative/replace -- always
            the full text, never a delta; spec ST.4.3).
        stable_until: Frozen-prefix length in codepoints; ``text[:stable_until]``
            is the part the engine has frozen and will not change (spec ST.4.2).
        state: Lifecycle as the UI sees it (``open`` / ``final`` / ``closed``).
        start: Segment start time in seconds (origin = first session sample).
        end: Segment end time in seconds.
        just_superseded: ``True`` for exactly one render after this segment was
            retired by a ``supersede``, so the renderer can briefly highlight the
            correction before the segment is dropped. The reducer sets it on the
            retired segments and the caller clears it after rendering.
    """

    segment_id: str
    text: str = ""
    stable_until: int = 0
    state: SegmentState = "open"
    start: float | None = None
    end: float | None = None
    just_superseded: bool = False

    @property
    def stable_text(self) -> str:
        """The frozen prefix ``text[:stable_until]`` (defensively bounded).

        Returns:
            The frozen prefix, or ``""`` if nothing is validly frozen.
        """
        if self.stable_until <= 0:
            return ""
        return self.text[: self.stable_until]

    @property
    def unstable_text(self) -> str:
        """The not-yet-frozen tail ``text[stable_until:]``.

        Returns:
            The unstable suffix of the segment text.
        """
        if self.stable_until <= 0:
            return self.text
        return self.text[self.stable_until :]


@dataclass
class LiveTranscript:
    """Pure event -> view reducer for a streaming transcription session.

    Feed events with :meth:`apply`; read the rendered state off the public
    fields / helpers. Holds no engine, audio, or terminal reference.

    Args:
        order: Live segment ids in arrival (reading) order.
        segments: Map of segment id to its :class:`SegmentView`.
        retired: Segments just retired by a ``supersede``, kept for one render so
            the correction can be highlighted, then dropped on the next
            :meth:`apply` (or :meth:`drain_retired`).
        audio_processed_until: Latest monotonic audio-time cursor (seconds).
        detected_language: Latest detected language (BCP-47), if any engine event
            carried one.
        counts: Per event-type tally for the status panel.
        reconnecting: ``True`` once a ``progress(reconnect=True)`` arrives.
        last_gap: The most recent reconnect ``(gap_start, gap_end)``, if any.
        recoverable_errors: Non-terminal ``error`` events (e.g. ``content_lost``)
            for the warning banner; the stream continues after them.
        terminal: The terminal event (``done`` or a non-recoverable ``error``)
            that ended the session, or ``None`` while live.
    """

    order: list[str] = field(default_factory=list)
    segments: dict[str, SegmentView] = field(default_factory=dict)
    retired: list[SegmentView] = field(default_factory=list)
    audio_processed_until: float = 0.0
    detected_language: str | None = None
    counts: Counter[str] = field(default_factory=Counter)
    reconnecting: bool = False
    last_gap: tuple[float | None, float | None] | None = None
    recoverable_errors: list[TranscriptionEvent] = field(default_factory=list)
    terminal: TranscriptionEvent | None = None

    # ------------------------------------------------------------------ #
    # The reduce.
    # ------------------------------------------------------------------ #
    def apply(self, event: TranscriptionEvent) -> None:
        """Fold one event into the view state (the canonical reduce, ST.5.2).

        Mirrors the spec's required application reduce exactly, plus view-only
        bookkeeping (counts, frozen-prefix boundary, reconnect/error banners,
        terminal capture). Every compliant app MUST implement the ``partial`` /
        ``final`` / ``supersede`` branches; the rest drive the UI.

        Args:
            event: The next streaming event from the session. The event is
                already structurally validated by the protocol (e.g. ``supersede``
                ``old_ids``/``new_ids`` are disjoint and non-repeating), so the
                reducer trusts it rather than re-checking invariants.

        Returns:
            None.
        """
        # A new content event (or any event) supersedes last frame's highlight of
        # retired segments: clear it so the highlight shows for exactly one render.
        self.retired.clear()
        self.counts[event.type] += 1
        if event.detected_language is not None:
            self.detected_language = event.detected_language
        if event.audio_processed_until is not None:
            # The protocol guarantees a monotonic cursor; max() is belt-and-braces.
            self.audio_processed_until = max(
                self.audio_processed_until, event.audio_processed_until
            )

        handler = {
            "partial": self._apply_partial,
            "final": self._apply_final,
            "supersede": self._apply_supersede,
            "progress": self._apply_progress,
            "error": self._apply_error,
            "done": self._apply_done,
        }[event.type]
        handler(event)

    def _apply_partial(self, event: TranscriptionEvent) -> None:
        """Show a segment's current best guess (text may still change)."""
        assert event.segment_id is not None  # protocol guarantees it for partial
        seg = self._upsert(event.segment_id)
        seg.text = event.text or ""
        seg.stable_until = self._bounded_stable_until(event, seg)
        seg.state = "open"
        self._set_times(seg, event)

    def _apply_final(self, event: TranscriptionEvent) -> None:
        """Commit a segment. Text is REPLACED, never appended.

        A ``closed`` final (``finality == "closed"``) may rewrite or even shorten
        the text (post-processing punctuation / ITN); the reducer replaces the
        displayed text accordingly (spec ST.5.3 / ST.5.4).
        """
        assert event.segment_id is not None  # protocol guarantees it for final
        seg = self._upsert(event.segment_id)
        seg.text = event.text or ""
        seg.stable_until = self._bounded_stable_until(event, seg)
        seg.state = "closed" if event.finality == "closed" else "final"
        self._set_times(seg, event)

    def _apply_supersede(self, event: TranscriptionEvent) -> None:
        """Replace a group of old segments with new ones (the must-have).

        The retired ``old_ids`` are removed from the live view (and stashed in
        :attr:`retired` for a one-frame highlight). The ``new_ids`` segments are
        NOT created here -- they arrive via their own ``partial`` / ``final``
        events, which is exactly when the reducer renders the correction live.
        """
        for old_id in event.old_ids:
            seg = self.segments.pop(old_id, None)
            if seg is not None:
                seg.just_superseded = True
                self.retired.append(seg)
                if old_id in self.order:
                    self.order.remove(old_id)
        # new_ids intentionally not pre-created: their first partial/final upserts
        # them, so the new text streams in visibly rather than popping in blank.

    def _apply_progress(self, event: TranscriptionEvent) -> None:
        """Advance the cursor; surface a reconnect banner if this is one."""
        if event.reconnect:
            self.reconnecting = True
            self.last_gap = (event.gap_start, event.gap_end)

    def _apply_error(self, event: TranscriptionEvent) -> None:
        """Record an error: recoverable -> banner + continue; terminal -> end."""
        if event.is_terminal:
            self.terminal = event
        else:
            # A recoverable error (e.g. content_lost) is a fidelity warning; the
            # session continues. A reconnect that recovered also clears the banner.
            self.recoverable_errors.append(event)
            self.reconnecting = False

    def _apply_done(self, event: TranscriptionEvent) -> None:
        """Mark the session finished."""
        self.terminal = event
        self.reconnecting = False

    # ------------------------------------------------------------------ #
    # Helpers.
    # ------------------------------------------------------------------ #
    def _upsert(self, segment_id: str) -> SegmentView:
        """Return the segment view for ``segment_id``, creating it if new.

        Args:
            segment_id: The segment id to fetch or create.

        Returns:
            The (possibly newly created and appended) segment view.
        """
        seg = self.segments.get(segment_id)
        if seg is None:
            seg = SegmentView(segment_id=segment_id)
            self.segments[segment_id] = seg
            self.order.append(segment_id)
        return seg

    @staticmethod
    def _bounded_stable_until(event: TranscriptionEvent, seg: SegmentView) -> int:
        """Clamp ``stable_until`` to a valid codepoint prefix of the text.

        The protocol already guards monotonicity and the combining-character
        boundary at the session layer, and rejects an out-of-range value at event
        construction. This is a thin defensive bound for the renderer so a None
        (engine reported no frozen prefix) becomes ``0`` and the value can never
        index past the text.

        Args:
            event: The partial/final event.
            seg: The segment view being updated (carries the new ``text``).

        Returns:
            A frozen-prefix length in ``[0, len(text)]``.
        """
        if event.stable_until is None:
            return 0
        return max(0, min(event.stable_until, len(seg.text)))

    @staticmethod
    def _set_times(seg: SegmentView, event: TranscriptionEvent) -> None:
        """Copy segment start/end times from the event when present.

        Args:
            seg: The segment view to update.
            event: The source event.

        Returns:
            None.
        """
        if event.start is not None:
            seg.start = event.start
        if event.end is not None:
            seg.end = event.end

    def drain_retired(self) -> list[SegmentView]:
        """Return and clear the segments retired since the last render.

        Returns:
            The just-superseded segments to highlight once, then forget.
        """
        retired = self.retired
        self.retired = []
        return retired

    # ------------------------------------------------------------------ #
    # Projections for the renderer / final output.
    # ------------------------------------------------------------------ #
    def live_segments(self) -> list[SegmentView]:
        """Return the live segments in reading order (excludes retired).

        Returns:
            The current segment views, ordered as they should be displayed.
        """
        return [self.segments[sid] for sid in self.order]

    def committed_text(self) -> str:
        """Return the joined text of committed (``final`` / ``closed``) segments.

        This is the view's notion of the final transcript, used to cross-check
        against the authoritative ``session.result().text`` at ``done``. Open
        (still-``partial``) segments are excluded -- they are not committed yet.

        Returns:
            The committed segments' text joined by single spaces.
        """
        parts = [
            self.segments[sid].text.strip()
            for sid in self.order
            if self.segments[sid].state in ("final", "closed")
        ]
        return " ".join(p for p in parts if p).strip()

    def is_finished(self) -> bool:
        """Whether a terminal event has ended the session.

        Returns:
            ``True`` once ``done`` or a non-recoverable ``error`` has arrived.
        """
        return self.terminal is not None

    @property
    def ended_in_error(self) -> bool:
        """Whether the session ended with a terminal error (not a clean done).

        Returns:
            ``True`` if the terminal event is an ``error``.
        """
        return self.terminal is not None and self.terminal.type == "error"


__all__ = ["LiveTranscript", "SegmentState", "SegmentView"]
