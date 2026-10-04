# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the event -> view reducer (LiveTranscript).

This is the most important test module in the repo: it proves the reducer folds
every streaming event type correctly -- especially ``supersede`` corrections and
each segment's growing ``stable_text`` -- by feeding scripted event lists
and asserting the resulting state exactly. No engine, audio, or terminal is
involved, so the tests are deterministic and fast.
"""

from __future__ import annotations

from standard_asr import TranscriptionEvent
from standard_asr.runtime.streaming import reduce_event as protocol_reduce

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
            TranscriptionEvent.partial("s0", "hello wor"),
            TranscriptionEvent.partial("s0", "hello world", stable_text="hello "),
            TranscriptionEvent.final("s0", "hello world."),
        ]
    )
    segs = state.live_segments()
    assert len(segs) == 1
    seg = segs[0]
    assert seg.text == "hello world."
    assert seg.state == "final"
    assert seg.stable_text == "hello world."
    assert state.counts["partial"] == 2
    assert state.counts["final"] == 1


def test_partial_stable_text_split() -> None:
    """The stable text / unstable rest split is exposed for rendering."""
    state = _drive([TranscriptionEvent.partial("s0", "the quick brown", stable_text="the ")])
    seg = state.live_segments()[0]
    assert seg.stable_text == "the "
    assert seg.unstable_text == "quick brown"


def test_final_text_is_replaced_not_appended() -> None:
    """A ``closed`` final REPLACES the displayed text (post-processing rewrite)."""
    state = _drive(
        [
            TranscriptionEvent.partial("s0", "twenty twenty"),
            TranscriptionEvent.closed("s0", "2020"),
        ]
    )
    seg = state.live_segments()[0]
    assert seg.text == "2020"  # replaced, not "twenty twenty2020"
    assert seg.state == "closed"


def test_partial_without_stable_text_shows_nothing_stable() -> None:
    """A ``partial`` that leaves ``stable_text`` out renders nothing as stable."""
    state = _drive([TranscriptionEvent.partial("s0", "anything")])
    seg = state.live_segments()[0]
    assert seg.stable_text == ""
    assert seg.unstable_text == "anything"


def test_final_is_stable_as_a_whole() -> None:
    """A ``final`` carries its whole text as stable text; nothing is left unstable."""
    state = _drive([TranscriptionEvent.final("s0", "abc")])
    seg = state.live_segments()[0]
    assert seg.stable_text == "abc"
    assert seg.unstable_text == ""


def test_unstable_text_removes_exactly_the_stable_code_points() -> None:
    """The rest of the text starts right after the stable text's code points.

    The stable text ends after ``e`` + U+0301 (two code points, one
    user-perceived character), so a split by user-perceived characters would
    cut in the wrong place.
    """
    text = "cafe\u0301 au lait"
    state = _drive([TranscriptionEvent.partial("s0", text, stable_text="cafe\u0301 ")])
    seg = state.live_segments()[0]
    assert seg.stable_text == "cafe\u0301 "
    assert seg.unstable_text == "au lait"
    assert seg.stable_text + seg.unstable_text == text


# --------------------------------------------------------------------------- #
# supersede (the must-have)
# --------------------------------------------------------------------------- #
def test_supersede_removes_old_and_renders_new() -> None:
    """supersede retires old segments; new ones appear via their own events."""
    state = _drive(
        [
            TranscriptionEvent.final("s0", "the quick brown fox"),
            TranscriptionEvent.supersede(old_ids=["s0"], new_ids=["s1", "s2"]),
            TranscriptionEvent.final("s1", "the quick "),
            TranscriptionEvent.final("s2", "brown fox jumps"),
        ]
    )
    ids = [s.segment_id for s in state.live_segments()]
    assert ids == ["s1", "s2"]  # s0 gone, replacements in reading order
    assert "s0" not in state.segments
    assert state.counts["supersede"] == 1


def test_supersede_marks_retired_for_one_highlight_frame() -> None:
    """A retired segment is exposed once (highlight), then dropped."""
    state = LiveTranscript()
    state.apply(TranscriptionEvent.final("s0", "old text"))
    state.apply(TranscriptionEvent.supersede(old_ids=["s0"], new_ids=["s1"]))
    # Immediately after the supersede, the retired segment is available to render.
    retired = list(state.retired)
    assert [s.segment_id for s in retired] == ["s0"]
    assert retired[0].just_superseded is True
    # The next event clears the one-frame highlight.
    state.apply(TranscriptionEvent.partial("s1", "new text"))
    assert state.retired == []


def test_supersede_merge_many_to_one() -> None:
    """supersede handles a many->one merge (two finals replaced by one)."""
    state = _drive(
        [
            TranscriptionEvent.final("s0", "hello"),
            TranscriptionEvent.final("s1", "world"),
            TranscriptionEvent.supersede(old_ids=["s0", "s1"], new_ids=["s2"]),
            TranscriptionEvent.final("s2", "hello world"),
        ]
    )
    ids = [s.segment_id for s in state.live_segments()]
    assert ids == ["s2"]
    assert state.live_segments()[0].text == "hello world"


def test_supersede_withdraws_retired_stable_text() -> None:
    """A supersede withdraws the retired segment, its stable text included.

    The replacement starts over with no stable text, and its text need not
    repeat what the retired segment had marked stable.
    """
    state = _drive(
        [
            TranscriptionEvent.partial("s0", "recognise speech", stable_text="recognise "),
            TranscriptionEvent.supersede(old_ids=["s0"], new_ids=["s1"]),
        ]
    )
    # Retired: gone from the live view; nothing of it is shown as stable.
    assert "s0" not in state.segments
    assert all(seg.stable_text == "" for seg in state.live_segments())
    state.apply(TranscriptionEvent.partial("s1", "wreck a nice beach"))
    seg = state.live_segments()[0]
    assert seg.segment_id == "s1"
    assert seg.stable_text == ""
    assert seg.unstable_text == "wreck a nice beach"


def test_supersede_places_replacements_where_the_retired_block_was() -> None:
    """Replacements take the retired block's reading position, not the end."""
    state = _drive(
        [
            TranscriptionEvent.final("a", "hi"),
            TranscriptionEvent.final("b", "world"),
            TranscriptionEvent.supersede(old_ids=["a"], new_ids=["a2"]),
            TranscriptionEvent.final("a2", "HI"),
        ]
    )
    assert [s.segment_id for s in state.live_segments()] == ["a2", "b"]
    assert state.committed_text() == "HI world"


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
            TranscriptionEvent.partial("s0", "hi", audio_processed_until=1.0),
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
            TranscriptionEvent.final("s0", "still going"),
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
    state = _drive([TranscriptionEvent.final("s0", "all done"), TranscriptionEvent.done()])
    assert state.is_finished() is True
    assert state.ended_in_error is False


def test_detected_language_tracked() -> None:
    """The detected language is captured from whichever event carries it."""
    state = _drive([TranscriptionEvent.final("s0", "bonjour", detected_language="fr")])
    assert state.detected_language == "fr"


# --------------------------------------------------------------------------- #
# committed text + cross-check against the protocol's own reduce
# --------------------------------------------------------------------------- #
def test_committed_text_excludes_open_partials() -> None:
    """committed_text() reflects only settled segments, not in-progress ones."""
    state = _drive(
        [
            TranscriptionEvent.final("s0", "first"),
            TranscriptionEvent.partial("s1", "second-in-progress"),
        ]
    )
    assert state.committed_text() == "first"


def test_matches_protocol_reduce_for_committed_map() -> None:
    """Our live segments agree with the protocol's canonical reduce_event.

    ``reduce_event`` is the spec's reference application reduce over a
    reading-order list plus a ``{segment_id: text}`` map. Driving the same
    events through it and through our reducer must agree on which segments
    survive, their text, and their reading order -- proving our richer view
    state still implements the canonical semantics. A segment after the
    retired one (``s9``) makes the order check see where the replacements go.
    """
    events = [
        TranscriptionEvent.partial("s0", "the quick brown fox", stable_text="the "),
        TranscriptionEvent.final("s0", "the quick brown fox"),
        TranscriptionEvent.final("s9", "over the lazy dog"),
        TranscriptionEvent.supersede(old_ids=["s0"], new_ids=["s1", "s2"]),
        TranscriptionEvent.final("s1", "the quick"),
        TranscriptionEvent.final("s2", "brown fox jumps"),
        TranscriptionEvent.done(),
    ]
    order: list[str] = []
    texts: dict[str, str] = {}
    for ev in events:
        protocol_reduce(order, texts, ev)

    state = _drive(events)
    assert [s.segment_id for s in state.live_segments()] == order
    assert {s.segment_id: s.text for s in state.live_segments()} == texts
