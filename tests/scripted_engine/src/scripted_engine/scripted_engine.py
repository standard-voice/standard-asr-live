# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""A scripted, fully compliant streaming Standard ASR engine for tests/demos.

Neither cookbook engine (``std-dummy-asr`` / ``std-faster-whisper``) implements
streaming -- both are batch-only. To prove that standard-asr-live's correction
rendering is correct against the *real* protocol types, this module provides a
streaming engine whose session yields a **deterministic, scripted** event
sequence exercising every behaviour the live UI must handle:

    partial -> partial -> final        (segment settles, stable_until grows)
    supersede (split one final into two new segments)
    partial -> final (each new segment)
    a recoverable error (content_lost-style) that the UI must survive
    done

It is a protocol-level test double, exactly like the cookbook dummy engine but
for streaming. The application never imports it: it is discovered via a
test-only entry point (see this package's ``pyproject.toml``) and driven through
the same ``discover_models`` -> ``supports`` -> ``start_transcription`` path the
app uses in production.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any, ClassVar, Literal

from standard_asr import (
    AudioFormat,
    RuntimeParams,
    TranscriptionEvent,
    TranscriptionResult,
)
from standard_asr.capabilities import (
    DeclaredCapabilities,
    FinalityCap,
    FlagCap,
    LanguageCaps,
    StreamingCapabilities,
)
from standard_asr.engine import (
    BaseConfig,
    BaseProperties,
    EngineBase,
    InputKind,
    LanguageConfigMixin,
    PreparedAudio,
)
from standard_asr.streaming import TranscriptionSession


class ScriptedConfig(LanguageConfigMixin, BaseConfig[Literal["scripted"]]):
    """Configuration for the scripted streaming engine.

    Args:
        engine: Discriminator (always ``"scripted"``).
        default_language: Default language for the engine.
    """

    engine: Literal["scripted"] = "scripted"
    default_language: str | None = "en"


class ScriptedProperties(BaseProperties):
    """Static metadata for the scripted streaming engine."""

    engine_id: str = "scripted"
    model_name: str = "demo"
    protocol_version: str = "1.0.0"
    accepted_input: set[InputKind] = {InputKind.ARRAY}
    native_sample_rate: int = 16000
    accepted_sample_rates: list[int] = [16000]
    # Declare the canonical wire encoding so incremental sessions are validated.
    wire_encodings: list[str] | None = ["pcm_s16le"]
    selectable_languages: list[str] = ["en", "auto"]
    detectable_languages: list[str] = ["en"]
    description: str | None = "Scripted streaming engine emitting partial/final/supersede."


#: A streaming engine declaring partials, re-segmentation (supersede) and a
#: meaningful stable_until -- exactly the capabilities the live UI gates on.
_CAPABILITIES = DeclaredCapabilities(
    streaming=StreamingCapabilities(
        language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        emits_partials=FlagCap(supported=True),
        re_segments=FlagCap(supported=True),
        word_stability=FlagCap(supported=True),
        finality_level=FinalityCap(mode="closed"),
    ),
    streaming_input=FlagCap(supported=True),
    streaming_output=FlagCap(supported=True),
)


#: The scripted event sequence (built fresh per session so each run is clean).
def _script() -> list[TranscriptionEvent]:
    """Return the deterministic scripted event list.

    The sequence settles one segment (with a growing frozen prefix), then
    ``supersede``s it into two segments that each settle, then emits a
    recoverable error, then ``done``. Every text is the full cumulative segment
    text (never a delta), and ``stable_until`` only ever grows per segment.

    The frozen-prefix arithmetic is deliberately spec-correct so the protocol's
    ``_LifecycleGuard`` admits every event with no suppression (spec ST.5.2: a
    supersede MUST preserve the concatenated frozen prefix of the retired
    segments). The retired ``seg-0`` freezes ``"the quick brown "`` (16
    codepoints, with the trailing space); the split's replacements re-freeze the
    SAME concatenation -- ``seg-1`` freezes ``"the quick "`` (10) and ``seg-2``
    freezes ``"brown "`` (6), so ``F_new == F_old``.

    Returns:
        The scripted events, in delivery order.
    """
    return [
        # Segment seg-0: best guess grows; frozen prefix advances 0 -> 4 -> 16.
        # "the " is frozen at su=4, then "the quick brown " (16) at the final.
        TranscriptionEvent.partial("seg-0", "the quik", stable_until=0, audio_processed_until=0.5),
        TranscriptionEvent.partial(
            "seg-0", "the quick brown", stable_until=4, audio_processed_until=1.0
        ),
        TranscriptionEvent.final(
            "seg-0",
            "the quick brown fox",
            stable_until=16,  # "the quick brown " (incl. trailing space)
            start=0.0,
            end=2.0,
            audio_processed_until=2.0,
        ),
        # Re-segmentation: split seg-0 into seg-1 + seg-2 (a two-pass rescoring
        # style correction). The concatenated frozen prefix MUST be preserved:
        # F_old = "the quick brown " (16). The replacements below re-freeze it.
        TranscriptionEvent.supersede(
            old_ids=["seg-0"], new_ids=["seg-1", "seg-2"], audio_processed_until=2.0
        ),
        TranscriptionEvent.partial("seg-1", "the quick", stable_until=4, audio_processed_until=2.2),
        TranscriptionEvent.final(
            "seg-1",
            "the quick ",
            stable_until=10,  # "the quick " (incl. trailing space) -> F_new part 1
            start=0.0,
            end=1.0,
            audio_processed_until=2.5,
        ),
        TranscriptionEvent.partial(
            "seg-2", "brown fox", stable_until=6, audio_processed_until=2.6
        ),
        TranscriptionEvent.final(
            "seg-2",
            "brown fox jumps",
            stable_until=15,  # full text; first 6 ("brown ") completes F_new == F_old
            start=1.0,
            end=2.5,
            audio_processed_until=2.8,
            detected_language="en",
        ),
        # A second segment arrives normally and is post-processed (closed adds
        # punctuation / capitalization -- the UI must REPLACE, not append).
        TranscriptionEvent.partial(
            "seg-3", "over the lazy dog", stable_until=8, audio_processed_until=3.5
        ),
        TranscriptionEvent.closed(
            "seg-3",
            "Over the lazy dog.",
            stable_until=18,
            start=2.5,
            end=4.0,
            audio_processed_until=4.0,
        ),
        # A non-terminal recoverable error: the UI shows a banner and continues.
        TranscriptionEvent.make_error(
            code="content_lost", recoverable=True, extra={"detail": "demo fidelity warning"}
        ),
        TranscriptionEvent.progress(audio_processed_until=4.0),
    ]


class _ScriptedSession(TranscriptionSession):
    """A streaming session that replays the scripted event sequence.

    Consumes (and discards) any fed audio so the full-duplex contract is honoured
    -- the live app feeds real PCM chunks -- while yielding the fixed script. The
    base class appends the terminal ``done`` after :meth:`_produce` returns.

    Events are emitted **interleaved with a small inter-event delay** so a
    consumer observes each ``partial`` before the segment's ``final`` arrives
    (without the delay the protocol's backpressure buffer would legitimately
    coalesce a pending partial into its same-segment final, and the live UI would
    only ever see settled text). The delay is configurable so tests can run it at
    ``0`` while the demo uses a visible cadence.
    """

    def __init__(self, events: list[TranscriptionEvent], *, step_delay: float = 0.0) -> None:
        """Initialize the scripted session.

        Args:
            events: The scripted events to replay.
            step_delay: Seconds to pause between scripted events so a consumer
                can observe interim partials before the final coalesces them.
        """
        super().__init__()
        self._events = events
        self._step_delay = step_delay

    async def _produce(self) -> AsyncIterator[TranscriptionEvent]:
        """Drain fed audio and yield the scripted events with a pacing delay.

        Spawns a background task to consume fed audio (honouring full-duplex and
        the liveness anchor) while the main coroutine emits the script, pausing
        ``step_delay`` between events so interim partials reach the consumer.

        Yields:
            The scripted transcription events, in order.
        """
        import asyncio

        async def _drain() -> None:
            async for _chunk in self.audio_chunks():
                pass

        drain_task = asyncio.ensure_future(_drain())
        try:
            for event in self._events:
                yield event
                if self._step_delay:
                    await asyncio.sleep(self._step_delay)
        finally:
            drain_task.cancel()


class ScriptedASR(EngineBase):
    """A compliant streaming engine that replays a scripted event sequence."""

    properties: ClassVar[BaseProperties] = ScriptedProperties()
    declared_capabilities: ClassVar[DeclaredCapabilities] = _CAPABILITIES
    config_type: ClassVar[type[BaseConfig[str]] | None] = ScriptedConfig

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the engine.

        Args:
            **kwargs: Configuration overrides for :class:`ScriptedConfig`.
        """
        self.config = ScriptedConfig.from_env("scripted", **kwargs)

    def _transcribe(self, prepared: PreparedAudio, params: RuntimeParams) -> TranscriptionResult:
        """Return a trivial batch result (the engine is primarily streaming).

        Args:
            prepared: Engine-ready audio.
            params: Gated runtime parameters.

        Returns:
            A minimal transcription result.
        """
        return TranscriptionResult(text="the quick brown fox", detected_language="en")

    def _start_transcription(
        self,
        *,
        gated_params: RuntimeParams,
        audio_format: AudioFormat | None,
        prepared_audio: PreparedAudio | None,
    ) -> TranscriptionSession:
        """Construct the scripted streaming session.

        Args:
            gated_params: The gated, frozen runtime parameters.
            audio_format: The wire format for incremental frames, if any.
            prepared_audio: Whole-input audio, if any.

        Returns:
            A :class:`_ScriptedSession` replaying the scripted events.
        """
        # The inter-event pacing is read from an environment variable so a demo
        # can make interim partials visible without the application (which is
        # engine-agnostic) needing any knowledge of this test double. Defaults to
        # 0 (no delay) for fast, deterministic tests.
        delay = float(os.environ.get("STD_SCRIPTED_STEP_DELAY", "0") or "0")
        return _ScriptedSession(_script(), step_delay=delay)


def create() -> ScriptedASR:
    """Entry-point factory for the ``scripted/demo`` model.

    Returns:
        A configured :class:`ScriptedASR` instance.
    """
    return ScriptedASR()


__all__ = ["ScriptedASR", "ScriptedConfig", "ScriptedProperties", "create"]
