# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for mode selection and the three driver paths.

Mode selection is the crux of "write once, run with any engine": the same app
drives a streaming, whole-input, or batch engine purely from declared
capabilities. These tests use tiny fake engines (compliant ``StandardASR``
shapes) so each path is exercised deterministically without real models.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest
from standard_asr import (
    BaseConfig,
    BaseProperties,
    EngineBase,
    InputKind,
    PreparedAudio,
    RuntimeParams,
    Segment,
    TranscriptionResult,
)
from standard_asr.capabilities import (
    BatchCapabilities,
    DeclaredCapabilities,
)

from standard_asr_live.audio_io import ChunkPlan
from standard_asr_live.driver import DriveConfig, Mode, Source, drive, select_mode
from standard_asr_live.errors import MicrophoneUnsupportedError


class _BatchProps(BaseProperties):
    engine_id: str = "fakebatch"
    model_name: str = "x"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY, InputKind.ENCODED_FILE}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] = [16000]
    selectable_languages: list[str] = ["en", "auto"]
    detectable_languages: list[str] = ["en"]


class _BatchConfig(BaseConfig):
    engine: str = "fakebatch"
    default_language: str | None = "en"


class _BatchEngine(EngineBase):
    """A batch-only fake engine returning two fixed segments."""

    properties = _BatchProps()
    declared_capabilities = DeclaredCapabilities(
        batch=BatchCapabilities(),
    )
    config_type = _BatchConfig

    def __init__(self) -> None:
        self.config = _BatchConfig()

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        return TranscriptionResult(
            text="hello world",
            detected_language="en",
            duration=2.0,
            segments=[
                Segment(start=0.0, end=1.0, text="hello"),
                Segment(start=1.0, end=2.0, text="world"),
            ],
        )


def _write_silence_wav(path: Path, seconds: float = 1.0) -> Path:
    """Write a tiny silent mono 16 kHz WAV.

    Args:
        path: Destination path.
        seconds: Duration in seconds.

    Returns:
        The written path.
    """
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


def test_select_mode_batch_engine() -> None:
    """A batch-only engine selects BATCH mode."""
    assert select_mode(_BatchEngine()) is Mode.BATCH


def test_resolve_audio_format_uses_canonical_pcm() -> None:
    """The wire format is canonical pcm_s16le mono at the chosen rate."""
    from standard_asr_live.driver import resolve_audio_format

    fmt = resolve_audio_format(_BatchEngine(), 16000)
    assert fmt.encoding == "pcm_s16le"
    assert fmt.sample_rate == 16000
    assert fmt.channels == 1


def test_resolve_audio_format_rejects_non_pcm_engine() -> None:
    """An engine constraining wire_encodings to exclude pcm_s16le fails loudly."""
    from standard_asr_live.driver import resolve_audio_format

    class _MulawProps(_BatchProps):
        wire_encodings: list[str] | None = ["mulaw"]

    class _MulawEngine(_BatchEngine):
        properties = _MulawProps()

    with pytest.raises(MicrophoneUnsupportedError, match="pcm_s16le"):
        resolve_audio_format(_MulawEngine(), 16000)


def test_batch_driver_synthesizes_final_events(tmp_path: Path) -> None:
    """The batch driver replays segments as final events and yields a result."""
    wav = _write_silence_wav(tmp_path / "a.wav")
    cfg = DriveConfig(
        source=Source.FILE,
        file_path=str(wav),
        plan=ChunkPlan(sample_rate=16000, paced=False),
        params=RuntimeParams(),
    )
    session = drive(_BatchEngine(), cfg)
    events = list(session.events)
    types = [e.type for e in events]
    # progress(detected_language) + two finals + done.
    assert types[-1] == "done"
    assert types.count("final") == 2
    assert session.result() is not None
    assert session.result().text == "hello world"


def test_batch_driver_rejects_mic() -> None:
    """A mic source against a batch engine fails loudly before any audio."""
    cfg = DriveConfig(
        source=Source.MIC,
        file_path=None,
        plan=ChunkPlan(sample_rate=16000),
        params=RuntimeParams(),
    )
    with pytest.raises(MicrophoneUnsupportedError):
        drive(_BatchEngine(), cfg)


def test_batch_driver_requires_file_path() -> None:
    """A file source with no path raises a clear error."""
    cfg = DriveConfig(
        source=Source.FILE,
        file_path=None,
        plan=ChunkPlan(sample_rate=16000),
        params=RuntimeParams(),
    )
    with pytest.raises(ValueError, match="file path is required"):
        drive(_BatchEngine(), cfg)


def test_batch_driver_no_segments_emits_whole_text(tmp_path: Path) -> None:
    """A batch result without segments still yields one final + a transcript."""

    class _NoSegEngine(_BatchEngine):
        def _transcribe(
            self, prepared: PreparedAudio, params: RuntimeParams
        ) -> TranscriptionResult:
            return TranscriptionResult(text="just one line", detected_language="en")

    wav = _write_silence_wav(tmp_path / "b.wav")
    cfg = DriveConfig(
        source=Source.FILE,
        file_path=str(wav),
        plan=ChunkPlan(sample_rate=16000, paced=False),
        params=RuntimeParams(),
    )
    session = drive(_NoSegEngine(), cfg)
    events = list(session.events)
    finals = [e for e in events if e.type == "final"]
    assert len(finals) == 1
    assert finals[0].text == "just one line"
