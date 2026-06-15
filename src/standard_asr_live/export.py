# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Export a transcription result to TXT / SRT / VTT.

Subtitle rendering uses the protocol's own :func:`standard_asr.to_srt` /
:func:`standard_asr.to_vtt`, so every compliant engine gets correct, identical
subtitle output for free (spec TR.6). Plain text is ``result.text``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from standard_asr import TranscriptionResult, to_srt, to_vtt

from .errors import LiveAppError


@dataclass(frozen=True, slots=True)
class ExportResult:
    """The files written by an export.

    Args:
        txt: Path to the plain-text transcript.
        srt: Path to the SRT subtitles.
        vtt: Path to the WebVTT subtitles.
    """

    txt: Path
    srt: Path
    vtt: Path

    def as_list(self) -> list[Path]:
        """Return the written paths as a list.

        Returns:
            ``[txt, srt, vtt]``.
        """
        return [self.txt, self.srt, self.vtt]


def export_result(
    result: TranscriptionResult, out_dir: str | Path, *, stem: str = "transcript"
) -> ExportResult:
    """Write a result to ``<out_dir>/<stem>.{txt,srt,vtt}``.

    Args:
        result: The transcription result to export.
        out_dir: Destination directory (created if missing).
        stem: Base filename without extension.

    Returns:
        An :class:`ExportResult` naming the three written files.

    Raises:
        LiveAppError: If the directory cannot be created or a file cannot be
            written (e.g. a bad path or insufficient permissions).
    """
    directory = Path(out_dir)
    txt_path = directory / f"{stem}.txt"
    srt_path = directory / f"{stem}.srt"
    vtt_path = directory / f"{stem}.vtt"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # Plain text ends with a trailing newline so the file is POSIX-clean.
        txt_path.write_text(result.text + ("\n" if not result.text.endswith("\n") else ""), "utf-8")
        srt_path.write_text(to_srt(result), "utf-8")
        vtt_path.write_text(to_vtt(result), "utf-8")
    except OSError as exc:
        raise LiveAppError(f"Could not export to {out_dir!r}: {exc}.") from exc
    return ExportResult(txt=txt_path, srt=srt_path, vtt=vtt_path)


__all__ = ["ExportResult", "export_result"]
