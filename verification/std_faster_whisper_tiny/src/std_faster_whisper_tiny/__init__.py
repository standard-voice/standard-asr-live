# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""A `faster-whisper/tiny` preset for fast local verification.

This is a thin engine-author-style preset: it subclasses the cookbook
faster-whisper engine and overrides only ``model_size`` (the upstream weights id)
and ``model_name`` (so ``properties.model_id`` matches the entry-point key). It
exists solely so the real-audio demo can run on CPU quickly (the ``tiny`` model
is a small download), without changing the cookbook. It is an installable plugin
discovered via an entry point -- the standard-asr-live application never imports
it; it only uses ``discover_models``.
"""

from __future__ import annotations

from typing import ClassVar

from standard_asr.engine import BaseProperties
from std_faster_whisper.std_asr_faster_whisper import (
    FasterWhisperASR,
    FasterWhisperProperties,
)


class TinyProperties(FasterWhisperProperties):
    """Static metadata for the ``faster-whisper/tiny`` preset."""

    model_name: str = "tiny"
    description: str | None = "faster-whisper tiny (fast, CPU-friendly verification preset)."


class TinyASR(FasterWhisperASR):
    """The ``faster-whisper/tiny`` preset (smallest, fastest Whisper variant)."""

    model_size: ClassVar[str] = "tiny"
    properties: ClassVar[BaseProperties] = TinyProperties()


def create() -> TinyASR:
    """Entry-point factory for the ``faster-whisper/tiny`` preset.

    Returns:
        A configured :class:`TinyASR` instance.
    """
    return TinyASR()


__all__ = ["TinyASR", "TinyProperties", "create"]
