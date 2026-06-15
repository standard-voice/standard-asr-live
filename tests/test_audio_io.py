# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for audio chunking and file decoding."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
import types
import wave
from pathlib import Path
from typing import Any

import pytest

from standard_asr_live.audio_io import (
    ChunkPlan,
    decode_file_to_pcm,
    iter_file_chunks,
    iter_mic_chunks,
    list_input_devices,
    pcm_duration_seconds,
)
from standard_asr_live.errors import LiveAppError


def test_chunk_plan_arithmetic() -> None:
    """The chunk plan derives frame/byte counts from rate and chunk_ms."""
    plan = ChunkPlan(sample_rate=16000, chunk_ms=100)
    assert plan.frames_per_chunk == 1600
    assert plan.bytes_per_chunk == 3200  # 1600 frames * 2 bytes (int16 mono)
    assert plan.chunk_seconds == pytest.approx(0.1)


def test_chunk_plan_minimum_one_frame() -> None:
    """A tiny chunk_ms still yields at least one frame per chunk."""
    plan = ChunkPlan(sample_rate=16000, chunk_ms=0)
    assert plan.frames_per_chunk == 1


def test_pcm_duration() -> None:
    """PCM duration is bytes / 2 / sample_rate."""
    assert pcm_duration_seconds(b"\x00\x00" * 16000, 16000) == pytest.approx(1.0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_decode_and_chunk_wav(tmp_path: Path) -> None:
    """A WAV file decodes to mono 16 kHz PCM and chunks to the expected size."""
    wav = tmp_path / "tone.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x01\x00" * 16000)  # 1 s
    pcm = decode_file_to_pcm(str(wav), 16000)
    assert pcm_duration_seconds(pcm, 16000) == pytest.approx(1.0, abs=0.05)

    plan = ChunkPlan(sample_rate=16000, chunk_ms=100, paced=False)
    chunks = list(iter_file_chunks(str(wav), plan))
    assert len(chunks) >= 9  # ~10 chunks of 100 ms
    assert all(len(c) <= plan.bytes_per_chunk for c in chunks)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_decode_resamples_stereo_to_mono(tmp_path: Path) -> None:
    """A 48 kHz stereo WAV is decoded down to mono 16 kHz."""
    wav = tmp_path / "stereo.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"\x02\x00\x03\x00" * 48000)  # 1 s stereo
    pcm = decode_file_to_pcm(str(wav), 16000)
    # 1 s at 16 kHz mono int16 = 32000 bytes (allow small encoder slack).
    assert pcm_duration_seconds(pcm, 16000) == pytest.approx(1.0, abs=0.1)


# --------------------------------------------------------------------------- #
# Microphone capture (fake sounddevice -- no hardware / PortAudio needed)
# --------------------------------------------------------------------------- #
_MIC_PLAN = ChunkPlan(sample_rate=16000, chunk_ms=10)  # 160 frames -> 320 bytes/chunk


def _make_fake_sd(
    *,
    devices: list[dict[str, Any]] | None = None,
    fail_open: bool = False,
    fail_read_after: int | None = None,
    overflow_reads: set[int] | None = None,
) -> types.ModuleType:
    """Build a fake ``sounddevice`` module with configurable behaviour.

    Args:
        devices: What ``query_devices()`` returns.
        fail_open: Raise when constructing ``RawInputStream`` (no mic / denied).
        fail_read_after: Raise from ``read()`` once this many reads have happened.
        overflow_reads: 1-based read indices that report an input overflow.

    Returns:
        A module object suitable for ``sys.modules["sounddevice"]``.
    """
    flagged = overflow_reads or set()

    class RawInputStream:
        def __init__(self, **_kwargs: Any) -> None:
            if fail_open:
                raise RuntimeError("could not open input device")
            self._reads = 0

        def __enter__(self) -> RawInputStream:
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

        def read(self, frames: int) -> tuple[bytes, bool]:
            self._reads += 1
            if fail_read_after is not None and self._reads > fail_read_after:
                raise RuntimeError("input device disconnected")
            return (b"\x01\x02" * frames, self._reads in flagged)

    mod = types.ModuleType("sounddevice")
    mod.RawInputStream = RawInputStream  # type: ignore[attr-defined]
    mod.query_devices = lambda: (devices or [])  # type: ignore[attr-defined]
    return mod


def test_mic_yields_chunks_and_stops_on_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture yields fixed-size pcm_s16le chunks and stops when the event is set."""
    monkeypatch.setitem(sys.modules, "sounddevice", _make_fake_sd())
    stop = threading.Event()
    chunks = []
    for i, chunk in enumerate(iter_mic_chunks(_MIC_PLAN, stop=stop)):
        chunks.append(chunk)
        if i >= 2:
            stop.set()
    assert len(chunks) == 3
    assert all(len(c) == _MIC_PLAN.bytes_per_chunk for c in chunks)


def test_mic_open_failure_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure to open the stream (no mic / denied permission) is actionable."""
    monkeypatch.setitem(sys.modules, "sounddevice", _make_fake_sd(fail_open=True))
    with pytest.raises(LiveAppError, match="microphone"):
        next(iter_mic_chunks(_MIC_PLAN, stop=threading.Event()))


def test_mic_midstream_failure_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A device failure mid-capture surfaces an actionable message, not a raw error."""
    monkeypatch.setitem(sys.modules, "sounddevice", _make_fake_sd(fail_read_after=2))
    with pytest.raises(LiveAppError, match="mid-stream"):
        for _ in iter_mic_chunks(_MIC_PLAN, stop=threading.Event()):
            pass


def test_mic_overflow_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Dropped audio (input overflow) is surfaced as a warning, never silent."""
    monkeypatch.setitem(sys.modules, "sounddevice", _make_fake_sd(overflow_reads={1}))
    stop = threading.Event()
    with caplog.at_level(logging.WARNING, logger="standard_asr_live.audio"):
        gen = iter_mic_chunks(_MIC_PLAN, stop=stop)
        next(gen)  # first read reports an overflow
        stop.set()
        gen.close()
    assert any("overflow" in r.message for r in caplog.records)


def test_list_input_devices_filters_to_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only devices with at least one input channel are listed."""
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        _make_fake_sd(
            devices=[
                {"name": "Speakers", "max_input_channels": 0},
                {"name": "USB Mic", "max_input_channels": 2},
            ]
        ),
    )
    assert list_input_devices() == [(1, "USB Mic", 2)]


# --------------------------------------------------------------------------- #
# File decode (ffmpeg) error paths
# --------------------------------------------------------------------------- #
def test_decode_missing_ffmpeg_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing ffmpeg binary produces an actionable install hint."""
    monkeypatch.setattr("standard_asr_live.audio_io.shutil.which", lambda _name: None)
    with pytest.raises(LiveAppError, match="ffmpeg"):
        decode_file_to_pcm("x.wav", 16000)


def test_decode_ffmpeg_failure_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ffmpeg decode failure is wrapped with its stderr detail."""
    monkeypatch.setattr("standard_asr_live.audio_io.shutil.which", lambda _name: "/usr/bin/ffmpeg")

    def _run(cmd: list[str], **_kwargs: object) -> object:
        raise subprocess.CalledProcessError(1, cmd, stderr=b"Invalid data found")

    monkeypatch.setattr("standard_asr_live.audio_io.subprocess.run", _run)
    with pytest.raises(LiveAppError, match="ffmpeg failed"):
        decode_file_to_pcm("x.wav", 16000)
