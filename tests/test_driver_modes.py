# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for mode selection and the three driver paths.

Mode selection is the crux of "write once, run with any engine": the same app
drives a streaming, whole-input, or batch engine purely from declared
capabilities. These tests use tiny fake engines (compliant ``StandardASR``
shapes) so each path is exercised deterministically without real models.
"""

from __future__ import annotations

import logging
import threading
import wave
from pathlib import Path

import pytest
from standard_asr import (
    RuntimeParams,
    Segment,
    TranscriptionResult,
    discover_models,
)
from standard_asr.capabilities import (
    BatchCapabilities,
    DeclaredCapabilities,
)
from standard_asr.engine import (
    BaseConfig,
    BaseProperties,
    EngineBase,
    InputKind,
    PreparedAudio,
)

import standard_asr_live.driver as driver_mod
from standard_asr_live.audio_io import ChunkPlan
from standard_asr_live.driver import DriveConfig, DriveSession, Mode, Source, drive, select_mode
from standard_asr_live.engine_view import LiveTranscript
from standard_asr_live.errors import MicrophoneUnsupportedError


# --------------------------------------------------------------------------- #
# MIC-1 async bridge: capture must run off the event loop and stop cleanly.
# --------------------------------------------------------------------------- #
async def test_aiter_chunk_source_yields_all_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The async bridge relays every chunk from the (blocking) sync source."""

    def _gen(_cfg: object, _stop: threading.Event):
        for i in range(3):
            yield bytes([i, i])

    monkeypatch.setattr(driver_mod, "_chunk_source", _gen)
    stop = threading.Event()
    chunks = [c async for c in driver_mod._aiter_chunk_source(object(), stop)]
    assert chunks == [b"\x00\x00", b"\x01\x01", b"\x02\x02"]


async def test_aiter_chunk_source_sets_stop_on_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """Closing the bridge sets the stop event so a live (mic-like) source ends.

    Regression guard: the bridge must NOT close the underlying generator from the
    loop thread (it may be mid-read on the executor thread -> "generator already
    executing"); it signals ``stop`` so the source exits and releases its stream.
    """

    def _gen(_cfg: object, stop: threading.Event):
        while not stop.is_set():
            yield b"\x00\x00"

    monkeypatch.setattr(driver_mod, "_chunk_source", _gen)
    stop = threading.Event()
    agen = driver_mod._aiter_chunk_source(object(), stop)
    assert await agen.__anext__() == b"\x00\x00"
    await agen.aclose()
    assert stop.is_set()


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


# --------------------------------------------------------------------------- #
# DriveSession.request_stop / close / _join_worker
# --------------------------------------------------------------------------- #
def test_drive_session_request_stop_sets_event_without_closing() -> None:
    """request_stop() sets the capture stop event but does NOT close the stream."""
    stop = threading.Event()
    closed: list[bool] = []

    def gen():
        try:
            yield "e1"
        finally:
            closed.append(True)

    g = gen()
    next(g)  # start the generator (now suspended at the yield)
    session = DriveSession(
        mode=Mode.INCREMENTAL, events=g, _result_holder=[], _diag_holder=[], _stop=stop
    )
    session.request_stop()
    assert stop.is_set()
    assert closed == []  # request_stop must NOT close the generator
    g.close()


def test_drive_session_request_stop_is_noop_without_stop() -> None:
    """request_stop() on a non-incremental session (no _stop) is a safe no-op."""
    session = DriveSession(mode=Mode.BATCH, events=iter([]), _result_holder=[], _diag_holder=[])
    session.request_stop()  # must not raise


def test_drive_session_close_is_idempotent() -> None:
    """close() closes the generator once and is safe to call again."""
    closed: list[bool] = []

    def gen():
        try:
            yield "e1"
        finally:
            closed.append(True)

    g = gen()
    next(g)
    session = DriveSession(
        mode=Mode.INCREMENTAL, events=g, _result_holder=[], _diag_holder=[], _stop=threading.Event()
    )
    session.close()
    session.close()  # idempotent: must not raise
    assert closed == [True]  # generator cleanup ran exactly once


def test_join_worker_warns_when_thread_stays_alive(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A worker still alive after the join timeout is surfaced as a WARNING, never silent."""
    monkeypatch.setattr(driver_mod, "_WORKER_JOIN_TIMEOUT", 0.01)

    class _StuckThread:
        name = "stuck-worker"

        def join(self, timeout: float | None = None) -> None:
            return

        def is_alive(self) -> bool:
            return True

    with caplog.at_level(logging.WARNING, logger="standard_asr_live.driver"):
        driver_mod._join_worker(_StuckThread())
    assert any("did not stop" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Incremental bridge, end to end through the REAL protocol session -- the path
# that regressed into a stall + un-interruptible Ctrl-C and had no coverage.
# --------------------------------------------------------------------------- #
_KEY = "scripted/demo"
_SILENCE = b"\x00\x00" * 1600  # 100 ms of 16 kHz mono silence


@pytest.mark.parametrize("use_sync", [False, True])
def test_incremental_drive_delivers_all_events_and_terminates(
    monkeypatch: pytest.MonkeyPatch, use_sync: bool
) -> None:
    """Driving the incremental bridge to completion yields the scripted event
    stream and stops on its own -- for BOTH the async default and the ``--sync``
    bridge.

    A finite chunk source stands in for a microphone that has been stopped: the
    session finalizes, the terminal event flows through, and the authoritative
    result is recovered. This exercises the consumer/feed bridge that regressed
    into a hang (the blocking event-queue put froze the loop and the
    timeout-less get swallowed Ctrl-C). If the bridge wedged, this test would
    hang rather than pass.
    """
    if _KEY not in discover_models().names():  # pragma: no cover - env without plugin
        pytest.skip("scripted/demo engine not installed")
    engine = discover_models().create(_KEY)
    # Replace the real (blocking) chunk source with a finite list of silence, so
    # no microphone or ffmpeg is needed and the source ends deterministically.
    monkeypatch.setattr(
        driver_mod, "_chunk_source", lambda cfg, stop: iter([_SILENCE, _SILENCE])
    )
    cfg = DriveConfig(
        source=Source.MIC,
        file_path=None,
        plan=ChunkPlan(sample_rate=16000, chunk_ms=100, paced=False),
        params=RuntimeParams(),
        use_sync_bridge=use_sync,
    )
    session = drive(engine, cfg)
    state = LiveTranscript()
    for event in session.events:
        state.apply(event)
    session.close()

    assert state.is_finished() is True
    assert state.ended_in_error is False
    assert state.counts["final"] >= 1
    result = session.result()
    assert result is not None
    assert "brown fox jumps" in result.text
