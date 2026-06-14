# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the event -> view reducer (LiveTranscript).

This is the most important test module in the repo: it proves the reducer folds
every streaming event type correctly -- especially ``supersede`` corrections and
the monotonic ``stable_until`` frozen prefix -- by feeding scripted event lists
and asserting the resulting state exactly. No engine, audio, or terminal is
involved, so the tests are deterministic and fast.
"""

from __future__ import annotations

from standard_asr import TranscriptionEvent
from standard_asr import reduce_event as protocol_reduce

from standard_asr_live.engine_view import LiveTranscript


def _drive(events: list[TranscriptionEvent]) -> LiveTranscript:
    """Apply a list of events to a fresh reducer.

    Args:
        events: The events to apply in order.

    Returns:
        The resulting :class:`LiveTranscript` state.
    """
    state = LiveTranscript()
    for event in events:
        state.apply(event)
    return state


# --------------------------------------------------------------------------- #
# partial / final basics
# --------------------------------------------------------------------------- #
def test_partial_then_final_settles_segment() -> None:
    """A segment shown as partial becomes final with replaced text."""
    state = _drive(
        [
            TranscriptionEvent.partial("s0", "hello wor", stable_until=0),
            TranscriptionEvent.partial("s0", "hello world", stable_until=6),
            TranscriptionEvent.final("s0", "hello world.", stable_until=12),
        ]
    )
    segs = state.live_segments()
    assert len(segs) == 1
    seg = segs[0]
    assert seg.text == "hello world."
    assert seg.state == "final"
    assert seg.stable_until == 12
    assert state.counts["partial"] == 2
    assert state.counts["final"] == 1


def test_partial_stable_prefix_split() -> None:
    """The frozen prefix / unstable tail split is exposed for rendering."""
    state = _drive([TranscriptionEvent.partial("s0", "the quick brown", stable_until=4)])
    seg = state.live_segments()[0]
    assert seg.stable_text == "the "
    assert seg.unstable_text == "quick brown"


def test_final_text_is_replaced_not_appended() -> None:
    """A ``closed`` final REPLACES the displayed text (post-processing rewrite)."""
    state = _drive(
        [
            TranscriptionEvent.partial("s0", "twenty twenty", stable_until=0),
            TranscriptionEvent.closed("s0", "2020", stable_until=4),
        ]
    )
    seg = state.live_segments()[0]
    assert seg.text == "2020"  # replaced, not "twenty twenty2020"
    assert seg.state == "closed"


def test_stable_until_none_is_zero() -> None:
    """A missing ``stable_until`` renders as nothing frozen."""
    state = _drive([TranscriptionEvent.partial("s0", "anything")])
    seg = state.live_segments()[0]
    assert seg.stable_until == 0
    assert seg.stable_text == ""
    assert seg.unstable_text == "anything"


def test_stable_until_is_bounded_to_text_length() -> None:
    """The reducer never indexes a frozen prefix past the text length."""
    # Construct an event then tamper would fail validation; instead use a short
    # text with an in-range value and confirm the bound helper holds at the edge.
    state = _drive([TranscriptionEvent.final("s0", "abc", stable_until=3)])
    seg = state.live_segments()[0]
    assert seg.stable_until == 3
    assert seg.stable_text == "abc"
    assert seg.unstable_text == ""


# --------------------------------------------------------------------------- #
# supersede (the must-have)
# --------------------------------------------------------------------------- #
def test_supersede_removes_old_and_renders_new() -> None:
    """supersede retires old segments; new ones appear via their own events."""
    state = _drive(
        [
            TranscriptionEvent.final("s0", "the quick brown fox", stable_until=16),
            TranscriptionEvent.supersede(old_ids=["s0"], new_ids=["s1", "s2"]),
            TranscriptionEvent.final("s1", "the quick ", stable_until=10),
            TranscriptionEvent.final("s2", "brown fox jumps", stable_until=15),
        ]
    )
    ids = [s.segment_id for s in state.live_segments()]
    assert ids == ["s1", "s2"]  # s0 gone, replacements in reading order
    assert "s0" not in state.segments
    assert state.counts["supersede"] == 1


def test_supersede_marks_retired_for_one_highlight_frame() -> None:
    """A retired segment is exposed once (highlight), then dropped."""
    state = LiveTranscript()
    state.apply(TranscriptionEvent.final("s0", "old text", stable_until=0))
    state.apply(TranscriptionEvent.supersede(old_ids=["s0"], new_ids=["s1"]))
    # Immediately after the supersede, the retired segment is available to render.
    retired = list(state.retired)
    assert [s.segment_id for s in retired] == ["s0"]
    assert retired[0].just_superseded is True
    # The next event clears the one-frame highlight.
    state.apply(TranscriptionEvent.partial("s1", "new text", stable_until=0))
    assert state.retired == []


def test_supersede_merge_many_to_one() -> None:
    """supersede handles a many->one merge (two finals replaced by one)."""
    state = _drive(
        [
            TranscriptionEvent.final("s0", "hello", stable_until=0),
            TranscriptionEvent.final("s1", "world", stable_until=0),
            TranscriptionEvent.supersede(old_ids=["s0", "s1"], new_ids=["s2"]),
            TranscriptionEvent.final("s2", "hello world", stable_until=0),
        ]
    )
    ids = [s.segment_id for s in state.live_segments()]
    assert ids == ["s2"]
    assert state.live_segments()[0].text == "hello world"


def test_supersede_of_unknown_id_is_harmless() -> None:
    """Retiring an id we never saw does not crash (defensive)."""
    state = _drive([TranscriptionEvent.supersede(old_ids=["ghost"], new_ids=["s1"])])
    assert state.live_segments() == []
    assert state.retired == []  # nothing real was retired


# --------------------------------------------------------------------------- #
# progress / error / done
# --------------------------------------------------------------------------- #
def test_progress_advances_audio_cursor() -> None:
    """A progress event advances the audio cursor without adding segments."""
    state = _drive(
        [
            TranscriptionEvent.partial("s0", "hi", stable_until=0, audio_processed_until=1.0),
            TranscriptionEvent.progress(audio_processed_until=2.5),
        ]
    )
    assert state.audio_processed_until == 2.5
    assert len(state.live_segments()) == 1


def test_reconnect_progress_sets_banner() -> None:
    """A reconnect progress event raises the reconnect banner with the gap."""
    state = _drive(
        [TranscriptionEvent.progress(reconnect=True, gap_start=1.0, gap_end=2.0)]
    )
    assert state.reconnecting is True
    assert state.last_gap == (1.0, 2.0)


def test_recoverable_error_is_banner_not_terminal() -> None:
    """A recoverable error is logged for a banner; the session continues."""
    state = _drive(
        [
            TranscriptionEvent.make_error(code="content_lost", recoverable=True),
            TranscriptionEvent.final("s0", "still going", stable_until=0),
        ]
    )
    assert [e.code for e in state.recoverable_errors] == ["content_lost"]
    assert state.is_finished() is False
    assert state.live_segments()[0].text == "still going"


def test_terminal_error_ends_session() -> None:
    """A non-recoverable error ends the session and flags the error state."""
    state = _drive([TranscriptionEvent.make_error(code="engine_error", recoverable=False)])
    assert state.is_finished() is True
    assert state.ended_in_error is True
    assert state.terminal is not None and state.terminal.code == "engine_error"


def test_done_ends_session_cleanly() -> None:
    """A done event ends the session without the error flag."""
    state = _drive(
        [TranscriptionEvent.final("s0", "all done", stable_until=0), TranscriptionEvent.done()]
    )
    assert state.is_finished() is True
    assert state.ended_in_error is False


def test_detected_language_tracked() -> None:
    """The detected language is captured from whichever event carries it."""
    state = _drive(
        [TranscriptionEvent.final("s0", "bonjour", stable_until=0, detected_language="fr")]
    )
    assert state.detected_language == "fr"


# --------------------------------------------------------------------------- #
# committed text + cross-check against the protocol's own reduce
# --------------------------------------------------------------------------- #
def test_committed_text_excludes_open_partials() -> None:
    """committed_text() reflects only settled segments, not in-progress ones."""
    state = _drive(
        [
            TranscriptionEvent.final("s0", "first", stable_until=0),
            TranscriptionEvent.partial("s1", "second-in-progress", stable_until=0),
        ]
    )
    assert state.committed_text() == "first"


def test_matches_protocol_reduce_for_committed_map() -> None:
    """Our live segment map agrees with the protocol's canonical reduce_event.

    ``standard_asr.reduce_event`` is the spec's reference application reduce over
    a ``{segment_id: text}`` map. Driving the same events through it and through
    our reducer must agree on which segments survive and their text -- proving our
    richer view state still implements the canonical semantics.
    """
    events = [
        TranscriptionEvent.partial("s0", "the quick brown fox", stable_until=0),
        TranscriptionEvent.final("s0", "the quick brown fox", stable_until=0),
        TranscriptionEvent.supersede(old_ids=["s0"], new_ids=["s1", "s2"]),
        TranscriptionEvent.final("s1", "the quick", stable_until=0),
        TranscriptionEvent.final("s2", "brown fox jumps", stable_until=0),
        TranscriptionEvent.done(),
    ]
    reference: dict[str, str] = {}
    for ev in events:
        protocol_reduce(reference, ev)

    state = _drive(events)
    ours = {s.segment_id: s.text for s in state.live_segments()}
    assert ours == reference
