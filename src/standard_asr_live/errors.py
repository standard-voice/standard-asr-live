# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Application-level errors and friendly rendering of protocol exceptions.

The app honours the protocol's philosophy: explicit over implicit, fail loudly,
never silently degrade. These helpers turn both our own usage errors and the
standard layer's exceptions into clear, actionable terminal messages -- without
ever echoing credentials.
"""

from __future__ import annotations


class LiveAppError(Exception):
    """A user-facing application error with an actionable message.

    Raised for situations the app detects itself (no plugins installed, a mic
    request against a non-streaming engine, a missing audio backend). The CLI
    catches it and prints the message without a traceback.
    """


class MicrophoneUnsupportedError(LiveAppError):
    """The selected engine cannot accept incremental microphone input.

    Per the protocol, an engine that does not declare ``streaming_input`` cannot
    consume a live, unbounded source. We fail loudly rather than silently
    pretending to stream (a silent wrong behaviour is the cardinal sin).
    """


class NoEngineSelectedError(LiveAppError):
    """No model key was provided and none could be chosen interactively."""


def describe_exception(exc: BaseException) -> str:
    """Return a concise, safe one-line description of an exception.

    Uses the exception's own message (the standard layer is careful never to
    embed secrets in its messages) and its type name, so a developer sees both
    *what* failed and *which* contracted error it was.

    Args:
        exc: The exception to describe.

    Returns:
        A ``"<TypeName>: <message>"`` string (message omitted if empty).
    """
    message = str(exc).strip()
    name = type(exc).__name__
    return f"{name}: {message}" if message else name


__all__ = [
    "LiveAppError",
    "MicrophoneUnsupportedError",
    "NoEngineSelectedError",
    "describe_exception",
]
