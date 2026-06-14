# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Test-only scripted streaming Standard ASR engine (a protocol test double)."""

from __future__ import annotations

from .scripted_engine import ScriptedASR, create

__all__ = ["ScriptedASR", "create"]
