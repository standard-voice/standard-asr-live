# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""CLI smoke tests: models / show, and a full run against the scripted engine.

The ``run`` test drives the centerpiece path end to end in plain (non-TTY) mode:
discover -> create the scripted streaming engine -> feed paced PCM -> reduce ->
render the final frame -> export. It proves the whole app works with a real
compliant streaming engine, including the supersede correction and export.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from standard_asr import discover_models

from standard_asr_live.cli import main

_SCRIPTED = "scripted/demo"


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
