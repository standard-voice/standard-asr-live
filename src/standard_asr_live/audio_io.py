# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Audio input: file decoding and microphone capture, as PCM wire chunks.

The streaming protocol feeds **raw PCM frames** declared once via an
:class:`~standard_asr.AudioFormat`. This module produces those frames in the
canonical wire encoding ``pcm_s16le`` (16-bit signed little-endian, mono) at a
chosen sample rate, from either:

* an **audio file** -- decoded and resampled with ``ffmpeg`` (always available
  in this environment), chunked, and yielded at a configurable real-time-like
  pace so the full live UI is exercised without a physical microphone; or
* the **microphone** -- captured with ``sounddevice`` as an ``int16`` mono
  stream and yielded chunk by chunk.

Producing the *same* chunk shape from both sources means the session driver and
the reducer never need to know which one is feeding them.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass

from .errors import LiveAppError

_log = logging.getLogger("standard_asr_live.audio")

#: Canonical wire encoding for the streaming session (spec §AI: 16-bit signed
#: little-endian PCM is the canonical wire encoding ``pcm_s16le``).
WIRE_ENCODING = "pcm_s16le"

#: Bytes per sample for ``pcm_s16le`` mono (one int16 per frame).
_BYTES_PER_FRAME = 2


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """The wire chunking parameters for a streaming session.

    Args:
        sample_rate: PCM sample rate in Hz.
        chunk_ms: Wire chunk size in milliseconds.
        speed: Real-time pacing multiplier for file input (1.0 = real time;
            higher = faster than real time). Ignored for the microphone.
        paced: Whether to sleep between chunks to approximate real time (file
            input only). ``False`` feeds as fast as possible (CI / stress).
    """

    sample_rate: int
    chunk_ms: int = 100
    speed: float = 1.0
    paced: bool = True

    @property
    def frames_per_chunk(self) -> int:
        """Number of PCM frames in one wire chunk.

        Returns:
            ``sample_rate * chunk_ms / 1000`` frames (at least 1).
        """
        return max(1, int(self.sample_rate * self.chunk_ms / 1000))

    @property
    def bytes_per_chunk(self) -> int:
        """Number of bytes in one wire chunk (mono ``pcm_s16le``).

        Returns:
            ``frames_per_chunk * 2`` bytes.
        """
        return self.frames_per_chunk * _BYTES_PER_FRAME

    @property
    def chunk_seconds(self) -> float:
        """Wall-clock seconds one chunk represents at 1.0x speed.

        Returns:
            The chunk duration in seconds.
        """
        return self.frames_per_chunk / self.sample_rate


def decode_file_to_pcm(path: str, sample_rate: int) -> bytes:
    """Decode an audio file to mono ``pcm_s16le`` at ``sample_rate`` via ffmpeg.

    Args:
        path: Path to the input audio file (any format ffmpeg can read).
        sample_rate: Target PCM sample rate in Hz.

    Returns:
        The full decoded PCM byte buffer (mono, 16-bit LE).

    Raises:
        LiveAppError: If ``ffmpeg`` is not on PATH, or decoding fails.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise LiveAppError(
            "ffmpeg was not found on PATH. Install ffmpeg to decode audio files "
            "(e.g. 'brew install ffmpeg'), or use --mic with raw capture."
        )
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-f",
        "s16le",  # raw PCM, signed 16-bit little-endian
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",  # mono
        "-ar",
        str(sample_rate),  # resample to the session rate
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError as exc:  # pragma: no cover - which() guarded above
        raise LiveAppError(f"Failed to launch ffmpeg: {exc}.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", "replace").strip() if exc.stderr else ""
        raise LiveAppError(
            f"ffmpeg failed to decode {path!r}: {detail or 'unknown error'}."
        ) from exc
    if not proc.stdout:
        raise LiveAppError(f"ffmpeg produced no audio for {path!r} (empty or unreadable file).")
    return proc.stdout


def pcm_duration_seconds(pcm: bytes, sample_rate: int) -> float:
    """Return the duration in seconds of a mono ``pcm_s16le`` buffer.

    Args:
        pcm: The PCM byte buffer.
        sample_rate: The PCM sample rate in Hz.

    Returns:
        The buffer duration in seconds.
    """
    return len(pcm) / _BYTES_PER_FRAME / sample_rate


def iter_file_chunks(path: str, plan: ChunkPlan) -> Iterator[bytes]:
    """Yield real-time-paced PCM chunks decoded from an audio file.

    Decodes the whole file up front, then yields fixed-size chunks. When
    ``plan.paced`` is set, sleeps between chunks so the feed approximates real
    time divided by ``plan.speed`` -- this is what lets the file path exercise the
    full live UI (partials settling, corrections re-rendering) as if spoken.

    Args:
        path: Path to the input audio file.
        plan: The chunking / pacing plan.

    Yields:
        Mono ``pcm_s16le`` byte chunks, in order.

    Raises:
        LiveAppError: If decoding fails (propagated from :func:`decode_file_to_pcm`).
    """
    pcm = decode_file_to_pcm(path, plan.sample_rate)
    step = plan.bytes_per_chunk
    sleep_for = (plan.chunk_seconds / plan.speed) if plan.paced and plan.speed > 0 else 0.0
    for start in range(0, len(pcm), step):
        chunk = pcm[start : start + step]
        if not chunk:
            break
        yield chunk
        if sleep_for:
            time.sleep(sleep_for)


def list_input_devices() -> list[tuple[int, str, int]]:
    """List available microphone input devices.

    Returns:
        A list of ``(index, name, max_input_channels)`` for each device that has
        at least one input channel.

    Raises:
        LiveAppError: If ``sounddevice`` (PortAudio) is unavailable.
    """
    sd = _import_sounddevice()
    devices: list[tuple[int, str, int]] = []
    for index, dev in enumerate(sd.query_devices()):
        max_in = int(dev.get("max_input_channels", 0))
        if max_in > 0:
            devices.append((index, str(dev.get("name", f"device {index}")), max_in))
    return devices


def iter_mic_chunks(
    plan: ChunkPlan, *, device: int | None = None, stop: object | None = None
) -> Iterator[bytes]:
    """Yield PCM chunks captured live from the microphone.

    Opens a mono ``int16`` input stream at ``plan.sample_rate`` and yields fixed
    ``plan.frames_per_chunk`` blocks as raw ``pcm_s16le`` bytes until ``stop`` is
    set (a ``threading.Event``) or the generator is closed.

    Args:
        plan: The chunking plan (mic ignores ``speed`` / ``paced``).
        device: Optional input device index (default: system default input).
        stop: Optional ``threading.Event``; capture stops once it is set.

    Yields:
        Mono ``pcm_s16le`` byte chunks captured from the microphone.

    Raises:
        LiveAppError: If ``sounddevice`` / PortAudio is unavailable or the input
            stream cannot be opened (e.g. no microphone, denied permission).
    """
    sd = _import_sounddevice()
    blocksize = plan.frames_per_chunk
    try:
        stream = sd.RawInputStream(
            samplerate=plan.sample_rate,
            blocksize=blocksize,
            device=device,
            channels=1,
            dtype="int16",
        )
    except Exception as exc:  # noqa: BLE001 - normalized to a user-facing error
        raise LiveAppError(
            f"Could not open the microphone input stream: {exc}. "
            "Check that a microphone is connected and the terminal has mic permission."
        ) from exc
    overflows = 0
    try:
        with stream:
            while stop is None or not stop.is_set():  # type: ignore[union-attr]
                try:
                    data, overflowed = stream.read(blocksize)
                except Exception as exc:  # noqa: BLE001 - normalized to a user-facing error
                    # A mid-stream PortAudio failure (device unplugged, sample-rate
                    # change, OS revoked the device) must surface as an actionable
                    # message, never a raw traceback or silent stop.
                    raise LiveAppError(
                        f"Microphone capture failed mid-stream: {exc}. The device may "
                        "have been disconnected or its OS permission revoked; reconnect "
                        "it (or pick another with --list-devices) and try again."
                    ) from exc
                if overflowed:
                    # Dropped audio is never silent: surfacing it is required so the
                    # operator knows the transcript may be incomplete (AGENTS.md:
                    # silent wrong results are the cardinal sin).
                    overflows += 1
                    _log.warning(
                        "microphone input overflow: capture could not keep up and audio "
                        "samples were dropped (the transcript may be incomplete)."
                    )
                # RawInputStream returns a buffer of raw bytes already in pcm_s16le.
                yield bytes(data)
    finally:
        if overflows:
            _log.warning(
                "microphone capture ended with %d input overflow(s); some audio was dropped.",
                overflows,
            )


def _import_sounddevice() -> object:
    """Import ``sounddevice`` lazily, with a friendly error if unavailable.

    Importing is deferred (and isolated here) because PortAudio is an optional
    system dependency: file mode must work even where the mic backend cannot
    load. The CLI only reaches this for ``--mic`` / ``--list-devices``.

    Returns:
        The imported ``sounddevice`` module.

    Raises:
        LiveAppError: If ``sounddevice`` or its PortAudio backend is unavailable.
    """
    try:
        import sounddevice as sd  # noqa: PLC0415 - intentional lazy import
    except OSError as exc:  # PortAudio shared library missing
        raise LiveAppError(
            f"The audio backend (PortAudio) could not be loaded: {exc}. "
            "Microphone capture is unavailable; use --file instead."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise LiveAppError(
            f"sounddevice is unavailable: {exc}. Use --file instead, or install "
            "the audio backend."
        ) from exc
    return sd


__all__ = [
    "WIRE_ENCODING",
    "ChunkPlan",
    "decode_file_to_pcm",
    "iter_file_chunks",
    "iter_mic_chunks",
    "list_input_devices",
    "pcm_duration_seconds",
]
