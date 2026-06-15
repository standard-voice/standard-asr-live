# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""CLI smoke tests: models / show, and a full run against the scripted engine.

The ``run`` test drives the centerpiece path end to end in plain (non-TTY) mode:
discover -> create the scripted streaming engine -> feed paced PCM -> reduce ->
render the final frame -> export. It proves the whole app works with a real
compliant streaming engine, including the supersede correction and export.
"""

from __future__ import annotations

import argparse
import signal
import sys
import types
from pathlib import Path

import pytest
from standard_asr import discover_models

from standard_asr_live.cli import (
    _consume_with_graceful_stop,
    _open_events_log,
    _resolve_source,
    _validate_device,
    _with_default_command,
    main,
)
from standard_asr_live.driver import Source
from standard_asr_live.errors import LiveAppError

_SCRIPTED = "scripted/demo"


def _run_args(**overrides: object) -> argparse.Namespace:
    """Build a minimal ``run`` args namespace for source/device unit tests.

    Args:
        **overrides: Fields to set (e.g. ``mic=True``, ``file="x.wav"``).

    Returns:
        An ``argparse.Namespace`` with mic/file/device defaults applied.
    """
    ns = argparse.Namespace(mic=False, file=None, device=None, audio=None)
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _fake_sounddevice(devices: list[dict[str, object]]) -> types.ModuleType:
    """Build a minimal fake ``sounddevice`` exposing ``query_devices``."""
    mod = types.ModuleType("sounddevice")
    mod.query_devices = lambda: devices  # type: ignore[attr-defined]
    return mod


def _has_scripted() -> bool:
    """Return whether the scripted engine plugin is installed.

    Returns:
        ``True`` if ``scripted/demo`` is discoverable.
    """
    return _SCRIPTED in discover_models().names()


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    """``--version`` prints the version and exits 0."""
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "standard-asr-live" in capsys.readouterr().out


def test_models_lists_engines(capsys: pytest.CaptureFixture[str]) -> None:
    """``models`` lists discovered engines."""
    code = main(["models"])
    out = capsys.readouterr().out
    assert code == 0
    assert "model key" in out
    # At least the dummy engine should be present in the dev environment.
    assert "dummy/echo" in out or "scripted/demo" in out


def test_show_unknown_model_errors(capsys: pytest.CaptureFixture[str]) -> None:
    """``show`` on an unknown model reports an error and exits non-zero."""
    code = main(["show", "no-such/model"])
    assert code == 1
    assert "not found" in capsys.readouterr().out


def test_show_scripted_engine(capsys: pytest.CaptureFixture[str]) -> None:
    """``show`` renders properties, capabilities, and config for an engine."""
    if not _has_scripted():
        pytest.skip("scripted engine not installed")
    code = main(["show", _SCRIPTED])
    out = capsys.readouterr().out
    assert code == 0
    assert "Properties" in out
    assert "streaming" in out  # capability tree rendered
    assert "default_language" in out  # config field rendered


def test_run_scripted_full_flow_with_export(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A full ``run`` against the scripted engine reduces, renders, and exports.

    Uses ``--no-pace`` (feed as fast as possible) and ``--plain`` (no live
    redraw, deterministic capture) so the test is fast and stable, and a tiny
    generated WAV as the file source.
    """
    if not _has_scripted():
        pytest.skip("scripted engine not installed")
    wav = _write_silence_wav(tmp_path / "silence.wav")
    export_dir = tmp_path / "out"
    code = main(
        [
            "run",
            _SCRIPTED,
            "--file",
            str(wav),
            "--no-pace",
            "--plain",
            "--no-prompt",
            "--export",
            str(export_dir),
            "--json-events",
            str(tmp_path / "events.jsonl"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    # The corrected transcript made it to the final result panel.
    assert "brown fox jumps" in out
    assert "Over the lazy dog." in out
    # Export produced the three files.
    assert (export_dir / "transcript.txt").exists()
    assert (export_dir / "transcript.srt").exists()
    assert (export_dir / "transcript.vtt").exists()
    # The JSONL event log captured the supersede event.
    log = (tmp_path / "events.jsonl").read_text("utf-8")
    assert '"supersede"' in log
    assert '"seg-1"' in log


def test_run_mic_against_batch_engine_fails_loudly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requesting --mic on a batch-only engine fails loudly (explicit > implicit)."""
    if "dummy/echo" not in discover_models().names():
        pytest.skip("dummy engine not installed")
    code = main(["run", "dummy/echo", "--mic", "--no-prompt", "--plain"])
    out = capsys.readouterr().out
    assert code == 1
    assert "streaming_input" in out or "microphone" in out.lower()


def _write_silence_wav(path: Path) -> Path:
    """Write a tiny mono 16 kHz silent WAV file for file-source tests.

    Args:
        path: Destination path.

    Returns:
        The written path.
    """
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)  # 1 s of silence
    return path


# --------------------------------------------------------------------------- #
# Input validation (actionable errors, no tracebacks)
# --------------------------------------------------------------------------- #
def test_resolve_source_defaults_to_mic() -> None:
    """With neither a file nor --mic, the source defaults to the microphone."""
    source, path = _resolve_source(_run_args())
    assert source is Source.MIC
    assert path is None


def test_resolve_source_positional_file(tmp_path: Path) -> None:
    """A positional audio path selects file input."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"\x00")
    source, path = _resolve_source(_run_args(audio=str(audio)))
    assert source is Source.FILE
    assert path == str(audio)


def test_resolve_source_mic_and_file_conflict(tmp_path: Path) -> None:
    """Passing both --mic and a file is a clear conflict, not a silent pick."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"\x00")
    with pytest.raises(LiveAppError, match="[Bb]oth"):
        _resolve_source(_run_args(mic=True, audio=str(audio)))


def test_resolve_source_missing_file_errors() -> None:
    """A nonexistent --file path is reported clearly."""
    with pytest.raises(LiveAppError, match="not found"):
        _resolve_source(_run_args(file="/no/such/audio.wav"))


def test_resolve_source_mic() -> None:
    """--mic resolves to the microphone source with no file path."""
    source, path = _resolve_source(_run_args(mic=True))
    assert source is Source.MIC
    assert path is None


def test_validate_device_rejects_unknown_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """An out-of-range --device index lists the valid devices instead of a vague error."""
    monkeypatch.setitem(
        sys.modules, "sounddevice", _fake_sounddevice([{"name": "Mic", "max_input_channels": 1}])
    )
    with pytest.raises(LiveAppError, match="Invalid --device"):
        _validate_device(999)
    _validate_device(0)  # a valid index does not raise


def test_run_unknown_set_field_exits_clean(capsys: pytest.CaptureFixture[str]) -> None:
    """A bad --set key exits 1 with a friendly message, never a traceback."""
    if not _has_scripted():
        pytest.skip("scripted engine not installed")
    code = main(["run", _SCRIPTED, "--set", "nope=1", "--plain", "--no-prompt"])
    assert code == 1
    assert "Unknown config field" in capsys.readouterr().out


def test_with_default_command_injects_run() -> None:
    """A bare invocation or one starting with a model/flag is treated as `run`."""
    assert _with_default_command([]) == ["run"]
    assert _with_default_command(["faster-whisper/tiny"]) == ["run", "faster-whisper/tiny"]
    assert _with_default_command(["faster-whisper/tiny", "a.wav"]) == [
        "run",
        "faster-whisper/tiny",
        "a.wav",
    ]
    assert _with_default_command(["--mic"]) == ["run", "--mic"]
    # Explicit subcommands and top-level flags are left untouched.
    assert _with_default_command(["models"]) == ["models"]
    assert _with_default_command(["run", "x"]) == ["run", "x"]
    assert _with_default_command(["--version"]) == ["--version"]
    assert _with_default_command(["-h"]) == ["-h"]
    assert _with_default_command(["--help"]) == ["--help"]


# --------------------------------------------------------------------------- #
# Ctrl-C handling (_consume_with_graceful_stop) + events-log error wrapping
# --------------------------------------------------------------------------- #
class _FakeSession:
    """Minimal DriveSession stand-in for _consume_with_graceful_stop tests."""

    def __init__(self, events: object) -> None:
        self.events = events
        self.stop_calls = 0
        self.closed = 0

    def request_stop(self) -> None:
        self.stop_calls += 1

    def close(self) -> None:
        self.closed += 1


def test_consume_non_graceful_completes_without_interrupt() -> None:
    """A finite source that ends on its own returns interrupted=False."""
    fake = _FakeSession(iter(["a", "b", "c"]))
    seen: list = []
    interrupted = _consume_with_graceful_stop(fake, seen.append, graceful=False)
    assert interrupted is False
    assert seen == ["a", "b", "c"]
    assert fake.stop_calls == 0


def test_consume_non_graceful_aborts_on_first_ctrl_c() -> None:
    """A finite (file/batch) source aborts on the FIRST Ctrl-C -- not swallowed."""

    def gen():
        yield "a"
        raise KeyboardInterrupt

    fake = _FakeSession(gen())
    seen: list = []
    interrupted = _consume_with_graceful_stop(fake, seen.append, graceful=False)
    assert interrupted is True
    assert seen == ["a"]  # aborted immediately; no further draining
    assert fake.stop_calls == 1


def test_consume_graceful_drains_after_first_ctrl_c(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mic graceful stop: the first Ctrl-C stops capture but KEEPS draining finals."""
    captured: dict = {}

    def fake_signal(_signum, handler):
        captured["handler"] = handler
        return signal.SIG_DFL  # a non-None "previous" handler to restore

    monkeypatch.setattr(signal, "signal", fake_signal)

    def gen():
        yield "partial"
        captured["handler"](signal.SIGINT, None)  # user presses Ctrl-C mid-stream
        yield "final"  # engine end-of-stream finalization still flows through
        yield "done"

    fake = _FakeSession(gen())
    seen: list = []
    interrupted = _consume_with_graceful_stop(fake, seen.append, graceful=True)
    assert interrupted is True
    assert seen == ["partial", "final", "done"]  # drained to completion (grey->white)
    assert fake.stop_calls == 1


def test_consume_graceful_second_ctrl_c_hard_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second Ctrl-C stops the drain immediately."""
    captured: dict = {}

    def fake_signal(_signum, handler):
        captured["handler"] = handler
        return signal.SIG_DFL

    monkeypatch.setattr(signal, "signal", fake_signal)

    def gen():
        yield "partial"
        captured["handler"](signal.SIGINT, None)  # first: stop + keep draining
        captured["handler"](signal.SIGINT, None)  # second: raise -> hard abort
        yield "never"

    fake = _FakeSession(gen())
    seen: list = []
    interrupted = _consume_with_graceful_stop(fake, seen.append, graceful=True)
    assert interrupted is True
    assert seen == ["partial"]  # stopped draining after the second Ctrl-C
    assert fake.stop_calls >= 1


def test_consume_graceful_falls_back_when_handler_uninstallable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off the main thread (signal.signal raises), a KeyboardInterrupt still aborts cleanly."""

    def boom(_signum, _handler):
        raise ValueError("signal only works in main thread")

    monkeypatch.setattr(signal, "signal", boom)

    def gen():
        yield "a"
        raise KeyboardInterrupt

    fake = _FakeSession(gen())
    seen: list = []
    interrupted = _consume_with_graceful_stop(fake, seen.append, graceful=True)
    assert interrupted is True
    assert seen == ["a"]
    assert fake.stop_calls == 1


def test_open_events_log_oserror_is_wrapped(tmp_path: Path) -> None:
    """An unopenable --json-events path surfaces as a clean LiveAppError."""
    blocker = tmp_path / "blocker"
    blocker.write_text("x")  # a FILE, so using it as a parent directory fails
    args = argparse.Namespace(json_events=str(blocker / "events.jsonl"))
    with pytest.raises(LiveAppError, match="--json-events"):
        _open_events_log(args)
