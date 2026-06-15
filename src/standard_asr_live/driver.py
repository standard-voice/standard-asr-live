# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Session driver: choose the transcription mode and pump events to the reducer.

This is the engine-agnostic glue between an audio source, a Standard ASR engine,
and the :class:`~standard_asr_live.engine_view.LiveTranscript` reducer. It picks
the right protocol path purely from declared capabilities:

* ``streaming_input``  -> **incremental streaming**: open an ``audio_format``
  session and ``feed`` PCM chunks (file or microphone). The centerpiece path.
* ``streaming_output`` -> **whole-input streaming**: hand the whole file to
  ``start_transcription(audio=...)`` and stream the result events (file only --
  a live mic cannot be a single whole input).
* neither -> **batch fallback**: ``transcribe`` the whole file, then synthesize
  a tiny event stream (``final`` per segment + ``done``) so the same reducer and
  renderer drive the screen. A live mic here fails loudly (you cannot batch an
  unbounded source).

Each driver yields :class:`~standard_asr.TranscriptionEvent` objects (real ones
from the session, or synthesized for batch). The caller folds them into the
reducer and renders -- so the UI code is identical across all three modes.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import AsyncIterator, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from standard_asr import (
    AudioFormat,
    AudioPath,
    RuntimeParams,
    StreamDeadlines,
    SyncSession,
    TranscriptionEvent,
)

from .audio_io import WIRE_ENCODING, ChunkPlan, iter_file_chunks, iter_mic_chunks
from .errors import MicrophoneUnsupportedError

if TYPE_CHECKING:
    from standard_asr import StandardASR, TranscriptionResult

_log = logging.getLogger("standard_asr_live.driver")

#: How long to wait for a background session thread to stop before abandoning it.
_WORKER_JOIN_TIMEOUT = 10.0


def _join_worker(worker: threading.Thread) -> None:
    """Join the background session thread, warning if it does not stop in time.

    A clean shutdown joins promptly; if the thread is still alive after the
    timeout (e.g. an engine wedged in a blocking call) we abandon the daemon
    thread but warn, so a subsequent empty result is explained rather than
    silent.

    Args:
        worker: The background session thread.
    """
    worker.join(timeout=_WORKER_JOIN_TIMEOUT)
    if worker.is_alive():
        _log.warning(
            "session worker %r did not stop within %.0fs; abandoning it -- the final "
            "result/diagnostics may be missing or incomplete.",
            worker.name,
            _WORKER_JOIN_TIMEOUT,
        )


class Mode(str, Enum):
    """The transcription mode selected from the engine's capabilities.

    Attributes:
        INCREMENTAL: ``streaming_input`` -- feed PCM chunks live.
        WHOLE_INPUT: ``streaming_output`` only -- stream a whole-file result.
        BATCH: neither -- batch transcribe with a synthesized event stream.
    """

    INCREMENTAL = "incremental"
    WHOLE_INPUT = "whole_input"
    BATCH = "batch"


class Source(str, Enum):
    """The audio source.

    Attributes:
        FILE: An audio file.
        MIC: The live microphone.
    """

    FILE = "file"
    MIC = "mic"


@dataclass(frozen=True, slots=True)
class DriveConfig:
    """Everything the driver needs to run a session.

    Args:
        source: File or microphone.
        file_path: Path to the audio file (required for ``Source.FILE``).
        plan: Wire chunking / pacing plan.
        params: Per-request runtime parameters (e.g. language).
        device: Microphone device index (``Source.MIC`` only).
        use_sync_bridge: Drive incremental streaming via :class:`SyncSession`
            instead of the async iterator (exercises the protocol's sync mirror).
        strict_lifecycle: Ask the session to RAISE on illegal lifecycle
            transitions instead of suppressing + diagnosing them.
    """

    source: Source
    file_path: str | None
    plan: ChunkPlan
    params: RuntimeParams
    device: int | None = None
    use_sync_bridge: bool = False
    strict_lifecycle: bool = False


def select_mode(engine: StandardASR) -> Mode:
    """Choose the transcription mode from an engine's declared capabilities.

    Args:
        engine: The engine instance to inspect via ``supports``.

    Returns:
        The selected :class:`Mode` (incremental > whole-input > batch).
    """
    if engine.supports("streaming_input"):
        return Mode.INCREMENTAL
    if engine.supports("streaming_output"):
        return Mode.WHOLE_INPUT
    return Mode.BATCH


@dataclass
class DriveSession:
    """A running drive: an event iterator plus deferred diagnostics/result.

    The event iterator is consumed by the caller (folded into the reducer). After
    it is exhausted, :meth:`diagnostics` and :meth:`result` expose the session's
    final state. For batch mode these come from the single ``transcribe`` call.

    Args:
        mode: The selected mode.
        events: The event iterator to consume.
    """

    mode: Mode
    events: Iterator[TranscriptionEvent]
    _result_holder: list[TranscriptionResult]
    _diag_holder: list[list]
    _stop: threading.Event | None = None

    def diagnostics(self) -> list:
        """Return the session diagnostics gathered after iteration.

        Returns:
            The accumulated diagnostics (gating / language / lifecycle), or an
            empty list if none.
        """
        return self._diag_holder[0] if self._diag_holder else []

    def result(self):  # -> TranscriptionResult | None
        """Return the authoritative final result after iteration, if any.

        Returns:
            The :class:`~standard_asr.TranscriptionResult`, or ``None`` if the
            session produced none.
        """
        return self._result_holder[0] if self._result_holder else None

    def request_stop(self) -> None:
        """Signal the source to stop feeding without tearing down the stream.

        Unlike :meth:`close`, this does NOT close the event iterator: it sets the
        capture stop event so a live (microphone) source ends, which lets the
        engine run its end-of-stream finalization and emit its terminal
        ``final`` / ``done`` events to a still-consuming caller. That is what makes
        a Ctrl-C stop *finalize* the on-screen transcript (grey -> white) instead
        of discarding the in-flight finalization. Idempotent; a no-op for a
        non-incremental session (which has no capture stop event).
        """
        if self._stop is not None:
            self._stop.set()

    def close(self) -> None:
        """Stop the drive and release resources; idempotent.

        Closes the underlying event generator. On the incremental streaming path
        that runs the generator's cleanup -- set the capture stop event, join the
        background session thread, and recover the final result / diagnostics the
        worker produced -- so this is the way to end an in-progress mic session
        (e.g. the user pressing Ctrl-C) while still recovering the partial
        transcript. Safe to call after normal exhaustion (a no-op). For the
        whole-input / batch paths there is no stop event to join on, so an early
        close simply abandons the background worker (a daemon thread reaped at
        process exit); the on-screen transcript already rendered is unaffected.
        """
        close = getattr(self.events, "close", None)
        if close is not None:
            close()


def drive(engine: StandardASR, cfg: DriveConfig) -> DriveSession:
    """Open the right session for ``engine`` and return a consumable drive.

    Args:
        engine: The engine instance.
        cfg: The drive configuration.

    Returns:
        A :class:`DriveSession` whose ``events`` iterator yields transcription
        events for the reducer.

    Raises:
        MicrophoneUnsupportedError: If a microphone source is requested but the
            engine cannot accept incremental input (no ``streaming_input``).
        ValueError: If a file path is required but missing.
    """
    mode = select_mode(engine)
    if cfg.source is Source.MIC and mode is not Mode.INCREMENTAL:
        raise MicrophoneUnsupportedError(
            f"Engine {engine.properties.engine_id!r} does not declare "
            "'streaming_input', so it cannot accept live microphone audio. "
            "Use --file with this engine, or pick an engine that supports "
            "incremental streaming input."
        )
    if cfg.source is Source.FILE and cfg.file_path is None:
        raise ValueError("A file path is required for file input.")

    if mode is Mode.INCREMENTAL:
        return _drive_incremental(engine, cfg)
    if mode is Mode.WHOLE_INPUT:
        return _drive_whole_input(engine, cfg)
    return _drive_batch(engine, cfg)


# --------------------------------------------------------------------------- #
# Incremental streaming (the centerpiece).
# --------------------------------------------------------------------------- #
def _chunk_source(cfg: DriveConfig, stop: threading.Event) -> Iterable[bytes]:
    """Return the PCM chunk iterable for the configured source.

    Args:
        cfg: The drive configuration.
        stop: Stop event for microphone capture.

    Returns:
        An iterable of mono ``pcm_s16le`` byte chunks.
    """
    if cfg.source is Source.MIC:
        return iter_mic_chunks(cfg.plan, device=cfg.device, stop=stop)
    assert cfg.file_path is not None
    return iter_file_chunks(cfg.file_path, cfg.plan)


async def _aiter_chunk_source(cfg: DriveConfig, stop: threading.Event) -> AsyncIterator[bytes]:
    """Async-adapt the (blocking) chunk source so it never blocks the event loop.

    The sync chunk sources block: the microphone does a blocking PortAudio
    ``read()`` and file pacing does ``time.sleep()``. Draining them directly on
    the protocol's asyncio loop (``session.feed`` does ``for chunk in source``)
    would freeze the loop for up to one chunk window per item, stalling the
    session's in-loop deadlines and delaying clean cancellation. We therefore
    pull each chunk on a dedicated single-thread executor and ``await`` it, so
    the blocking I/O happens off the loop and the loop stays responsive. The
    underlying generator is closed on exit so its ``with stream`` (mic) / pacing
    cleanup runs.

    Args:
        cfg: The drive configuration.
        stop: Stop event for microphone capture.

    Yields:
        Mono ``pcm_s16le`` byte chunks, off-loop.
    """
    sync_iter = iter(_chunk_source(cfg, stop))
    loop = asyncio.get_running_loop()
    # A single worker keeps all PortAudio reads on one consistent thread.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sasr-live-capture")
    _sentinel = object()
    try:
        while True:
            chunk = await loop.run_in_executor(executor, lambda: next(sync_iter, _sentinel))
            if chunk is _sentinel:
                break
            yield chunk  # type: ignore[misc]
    finally:
        # Signal the capture to stop rather than closing the generator from this
        # thread: on cancellation the executor thread may still be inside a
        # blocking read, and closing a generator that is executing on another
        # thread raises "generator already executing". The mic loop exits on
        # `stop` and closes its own stream; the in-flight read then completes and
        # the capture thread exits on its own.
        stop.set()
        executor.shutdown(wait=False)


def resolve_audio_format(engine: StandardASR, sample_rate: int) -> AudioFormat:
    """Build the wire ``AudioFormat`` to open an incremental session with.

    The app produces canonical ``pcm_s16le`` mono PCM (from ffmpeg / the mic), so
    it uses that encoding -- but it honours the engine's declared
    ``wire_encodings`` when present: if the engine constrains its encodings and
    does NOT accept ``pcm_s16le``, we fail loudly here rather than open a session
    the engine will reject (or, worse, mis-frame). When ``wire_encodings`` is
    ``None`` (unconstrained / self-managed wire format) we send ``pcm_s16le``.

    This mirrors the standard layer's own (private) ``_streaming_audio_format``
    helper; see the findings doc -- exposing that publicly would let apps skip
    this entirely.

    Args:
        engine: The engine instance.
        sample_rate: The chosen PCM sample rate in Hz.

    Returns:
        A mono ``AudioFormat`` for :meth:`start_transcription`.

    Raises:
        MicrophoneUnsupportedError: If the engine declares ``wire_encodings`` that
            exclude the canonical ``pcm_s16le`` the app can produce.
    """
    declared = getattr(engine.properties, "wire_encodings", None)
    if declared is not None and WIRE_ENCODING not in declared:
        raise MicrophoneUnsupportedError(
            f"Engine {engine.properties.engine_id!r} declares wire_encodings="
            f"{declared} and does not accept the canonical {WIRE_ENCODING!r} PCM "
            "this app produces. A non-PCM wire codec is not supported by this demo."
        )
    return AudioFormat(encoding=WIRE_ENCODING, sample_rate=sample_rate, channels=1)


def _drive_incremental(engine: StandardASR, cfg: DriveConfig) -> DriveSession:
    """Drive an incremental ``audio_format`` streaming session.

    Args:
        engine: The engine instance.
        cfg: The drive configuration.

    Returns:
        A :class:`DriveSession` over the live event stream.
    """
    audio_format = resolve_audio_format(engine, cfg.plan.sample_rate)
    result_holder: list = []
    diag_holder: list = []
    stop = threading.Event()

    if cfg.use_sync_bridge:
        events = _iter_sync_incremental(engine, cfg, audio_format, result_holder, diag_holder, stop)
    else:
        events = _iter_async_incremental(
            engine, cfg, audio_format, result_holder, diag_holder, stop
        )
    return DriveSession(
        mode=Mode.INCREMENTAL,
        events=events,
        _result_holder=result_holder,
        _diag_holder=diag_holder,
        _stop=stop,
    )


def _iter_sync_incremental(
    engine: StandardASR,
    cfg: DriveConfig,
    audio_format: AudioFormat,
    result_holder: list,
    diag_holder: list,
    stop: threading.Event,
) -> Iterator[TranscriptionEvent]:
    """Drive an incremental session through the synchronous bridge.

    Uses :class:`SyncSession`, which owns a background event loop, so the whole
    feed/consume cycle is synchronous. ``feed`` accepts a sync iterable of byte
    chunks; the bridge consumes it on its own loop while we iterate events.

    Args:
        engine: The engine instance.
        cfg: The drive configuration.
        audio_format: The locked wire format.
        result_holder: One-slot list to receive the final result.
        diag_holder: One-slot list to receive diagnostics.
        stop: Stop event for microphone capture.

    Yields:
        Transcription events from the session.
    """
    inner = engine.start_transcription(
        audio_format=audio_format,
        params=cfg.params,
        deadlines=_deadlines(),
    )
    if cfg.strict_lifecycle:  # pragma: no cover - exercised via async path in tests
        _enable_strict_lifecycle(inner)
    session = SyncSession(inner)
    try:
        with session:
            session.feed(_chunk_source(cfg, stop))
            yield from session
            result_holder.append(session.result())
            diag_holder.append(session.diagnostics())
    finally:
        stop.set()


def _iter_async_incremental(
    engine: StandardASR,
    cfg: DriveConfig,
    audio_format: AudioFormat,
    result_holder: list,
    diag_holder: list,
    stop: threading.Event,
) -> Iterator[TranscriptionEvent]:
    """Drive an incremental session on a private asyncio loop, yielded as a sync iterator.

    Runs the async session on a background event loop in a dedicated thread and
    hands events to the (synchronous) renderer via a thread-safe queue. This lets
    the rich ``Live`` renderer stay on the main thread while the protocol's async
    session runs unmodified.

    Args:
        engine: The engine instance.
        cfg: The drive configuration.
        audio_format: The locked wire format.
        result_holder: One-slot list to receive the final result.
        diag_holder: One-slot list to receive diagnostics.
        stop: Stop event for microphone capture.

    Yields:
        Transcription events from the session.
    """
    import asyncio

    event_q: queue.Queue = queue.Queue(maxsize=512)
    sentinel = object()
    error_holder: list[BaseException] = []

    async def _run() -> None:
        session = engine.start_transcription(
            audio_format=audio_format,
            params=cfg.params,
            deadlines=_deadlines(),
        )
        if cfg.strict_lifecycle:
            _enable_strict_lifecycle(session)
        async with session:
            session.feed(_aiter_chunk_source(cfg, stop))
            async for event in session:
                event_q.put(event)
            result_holder.append(session.result())
            diag_holder.append(session.diagnostics())

    def _thread() -> None:
        try:
            asyncio.run(_run())
        except BaseException as exc:  # noqa: BLE001 - surfaced to the consumer
            error_holder.append(exc)
        finally:
            event_q.put(sentinel)

    worker = threading.Thread(target=_thread, name="sasr-live-stream", daemon=True)
    worker.start()
    try:
        while True:
            item = event_q.get()
            if item is sentinel:
                break
            yield item
    finally:
        stop.set()
        _join_worker(worker)
    if error_holder:
        raise error_holder[0]


# --------------------------------------------------------------------------- #
# Whole-input streaming (streaming_output only).
# --------------------------------------------------------------------------- #
def _drive_whole_input(engine: StandardASR, cfg: DriveConfig) -> DriveSession:
    """Drive a whole-input ``start_transcription(audio=...)`` streaming session.

    Args:
        engine: The engine instance.
        cfg: The drive configuration.

    Returns:
        A :class:`DriveSession` over the streamed result events.
    """
    import asyncio

    assert cfg.file_path is not None
    result_holder: list = []
    diag_holder: list = []
    event_q: queue.Queue = queue.Queue(maxsize=512)
    sentinel = object()
    error_holder: list[BaseException] = []

    async def _run() -> None:
        session = engine.start_transcription(
            audio=AudioPath(cfg.file_path), params=cfg.params, deadlines=_deadlines()
        )
        async with session:
            async for event in session:
                event_q.put(event)
            result_holder.append(session.result())
            diag_holder.append(session.diagnostics())

    def _thread() -> None:
        try:
            asyncio.run(_run())
        except BaseException as exc:  # noqa: BLE001
            error_holder.append(exc)
        finally:
            event_q.put(sentinel)

    def _events() -> Iterator[TranscriptionEvent]:
        worker = threading.Thread(target=_thread, name="sasr-live-whole", daemon=True)
        worker.start()
        while True:
            item = event_q.get()
            if item is sentinel:
                break
            yield item
        _join_worker(worker)
        if error_holder:
            raise error_holder[0]

    return DriveSession(
        mode=Mode.WHOLE_INPUT,
        events=_events(),
        _result_holder=result_holder,
        _diag_holder=diag_holder,
    )


# --------------------------------------------------------------------------- #
# Batch fallback.
# --------------------------------------------------------------------------- #
def _drive_batch(engine: StandardASR, cfg: DriveConfig) -> DriveSession:
    """Drive a batch ``transcribe`` and synthesize an event stream for the UI.

    The whole file is transcribed in one call; the result's segments are replayed
    as ``final`` events (plus a terminal ``done``) so the same reducer/renderer
    drive the screen. The result is computed lazily on first iteration so the
    caller can show a spinner while it runs.

    Args:
        engine: The engine instance.
        cfg: The drive configuration.

    Returns:
        A :class:`DriveSession` over the synthesized event stream.
    """
    assert cfg.file_path is not None
    result_holder: list = []
    diag_holder: list = []

    def _events() -> Iterator[TranscriptionEvent]:
        result = engine.transcribe(AudioPath(cfg.file_path), cfg.params)
        result_holder.append(result)
        diag_holder.append(list(result.diagnostics))
        if result.detected_language is not None:
            yield TranscriptionEvent.progress(detected_language=result.detected_language)
        segments = result.segments or []
        if segments:
            for index, seg in enumerate(segments):
                yield TranscriptionEvent.final(
                    segment_id=f"seg-{index}",
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    audio_processed_until=seg.end,
                )
        elif result.text:
            # No segments: emit the whole transcript as one final segment.
            yield TranscriptionEvent.final(segment_id="seg-0", text=result.text)
        yield TranscriptionEvent.done()

    return DriveSession(
        mode=Mode.BATCH,
        events=_events(),
        _result_holder=result_holder,
        _diag_holder=diag_holder,
    )


# --------------------------------------------------------------------------- #
# Shared bits.
# --------------------------------------------------------------------------- #
def _deadlines() -> StreamDeadlines:
    """Return the session deadlines this app requests.

    A live, file-paced demo can sit idle briefly between chunks; the protocol's
    defaults (300 s done-timeout, idle/wall caps disabled) are already correct, so
    we pass an explicit unmodified :class:`StreamDeadlines` to make the choice
    visible rather than implicit.

    Returns:
        The default :class:`StreamDeadlines`.
    """
    return StreamDeadlines()


def _enable_strict_lifecycle(session: object) -> None:
    """Flip a session's lifecycle guard to strict (raise on illegal transitions).

    The protocol exposes ``strict_lifecycle`` as a session constructor argument,
    but the engine builds the session, so the app reaches the guard through the
    documented-private ``_guard`` once the session exists. This is a deliberate,
    isolated reach into a private (so the demo can showcase strict mode); if the
    attribute is absent on a third-party session it is simply skipped.

    Args:
        session: The streaming session whose guard to make strict.

    Returns:
        None.
    """
    guard = getattr(session, "_guard", None)
    if guard is not None:
        guard._strict = True  # noqa: SLF001 - documented private reach for a demo toggle


__all__ = [
    "DriveConfig",
    "DriveSession",
    "Mode",
    "Source",
    "drive",
    "resolve_audio_format",
    "select_mode",
]
