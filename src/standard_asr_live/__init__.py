# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""standard-asr-live: live streaming speech-to-text in the terminal.

A reference application for the Standard ASR protocol. It performs live streaming
transcription with real-time on-screen corrections (``partial`` -> ``final`` ->
``supersede``) from a microphone or an audio file, against *any* installed
Standard ASR engine. It depends only on the protocol (``standard-asr``) and is
engine-agnostic by construction: engines are discovered via entry points and
never imported directly.

The heart of the app is :class:`~standard_asr_live.engine_view.LiveTranscript`,
a pure event-to-view reducer that folds the protocol's streaming events into the
state the terminal renders.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
