# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for TXT / SRT / VTT export."""

from __future__ import annotations

from pathlib import Path

import pytest
from standard_asr import Segment, TranscriptionResult

from standard_asr_live.errors import LiveAppError
from standard_asr_live.export import export_result


def _result() -> TranscriptionResult:
    """Build a small segmented result for export.

    Returns:
        A two-segment English result.
    """
    return TranscriptionResult(
        text="Hello world. The quick brown fox.",
        detected_language="en",
        duration=4.0,
        segments=[
            Segment(start=0.0, end=2.0, text="Hello world."),
            Segment(start=2.0, end=4.0, text="The quick brown fox."),
        ],
    )


def test_export_writes_three_files(tmp_path: Path) -> None:
    """Export writes transcript.txt / .srt / .vtt into the directory."""
    out = export_result(_result(), tmp_path)
    assert out.txt.exists() and out.srt.exists() and out.vtt.exists()
    assert {p.name for p in out.as_list()} == {
        "transcript.txt",
        "transcript.srt",
        "transcript.vtt",
    }


def test_txt_is_plain_transcript(tmp_path: Path) -> None:
    """The TXT export is the plain transcript text with a trailing newline."""
    out = export_result(_result(), tmp_path)
    content = out.txt.read_text("utf-8")
    assert content == "Hello world. The quick brown fox.\n"


def test_srt_has_indexed_timed_cues(tmp_path: Path) -> None:
    """The SRT export carries indexed, timestamped cues from the segments."""
    out = export_result(_result(), tmp_path)
    srt = out.srt.read_text("utf-8")
    assert "1\n00:00:00,000 --> 00:00:02,000\nHello world." in srt
    assert "2\n00:00:02,000 --> 00:00:04,000\nThe quick brown fox." in srt


def test_vtt_has_header_and_cues(tmp_path: Path) -> None:
    """The VTT export starts with the WEBVTT header and has dotted timestamps."""
    out = export_result(_result(), tmp_path)
    vtt = out.vtt.read_text("utf-8")
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.000\nHello world." in vtt


def test_custom_stem(tmp_path: Path) -> None:
    """A custom stem renames all three output files."""
    out = export_result(_result(), tmp_path, stem="meeting")
    assert out.txt.name == "meeting.txt"
    assert out.srt.name == "meeting.srt"
    assert out.vtt.name == "meeting.vtt"


def test_export_oserror_is_wrapped(tmp_path: Path) -> None:
    """A failure to create/write the export directory surfaces as a clean LiveAppError."""
    blocker = tmp_path / "blocker"
    blocker.write_text("x")  # a FILE; using it as a parent dir makes mkdir fail
    with pytest.raises(LiveAppError, match="Could not export"):
        export_result(_result(), blocker / "out")
