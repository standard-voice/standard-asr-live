# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for audio chunking and file decoding."""

from __future__ import annotations

import shutil
import wave
from pathlib import Path

import pytest

from standard_asr_live.audio_io import (
    ChunkPlan,
    decode_file_to_pcm,
    iter_file_chunks,
    pcm_duration_seconds,
)


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
