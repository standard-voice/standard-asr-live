# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Thin, engine-agnostic helpers over the Standard ASR discovery API.

Everything here goes through the protocol's :func:`standard_asr.discover_models`
and reads class-level metadata (``properties`` / ``declared_capabilities`` /
config JSON Schema) **without instantiating** an engine -- so a credentialed
engine can still be listed and have its settings form rendered before the user
has supplied the credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from standard_asr import (
    BaseProperties,
    DeclaredCapabilities,
    ModelRegistry,
    ModelSpec,
    discover_models,
)


@dataclass(frozen=True, slots=True)
class StreamingProfile:
    """A compact summary of an engine's streaming capabilities for display.

    Args:
        streaming_input: Engine can accept audio incrementally (mic / live feed).
        streaming_output: Engine can emit results before all input arrives.
        emits_partials: Engine emits ``partial`` events (live interim text).
        re_segments: Engine may emit ``supersede`` events (corrections).
        word_stability: Engine provides a meaningful ``stable_until`` frontier.
        finality_mode: Strongest finality level (``"final"`` / ``"closed"``), or
            ``None`` if streaming is unsupported.
    """

    streaming_input: bool
    streaming_output: bool
    emits_partials: bool
    re_segments: bool
    word_stability: bool
    finality_mode: str | None

    @property
    def any_streaming(self) -> bool:
        """Whether the engine supports streaming at all (input or output).

        Returns:
            ``True`` if either streaming axis is supported.
        """
        return self.streaming_input or self.streaming_output

    def headline(self) -> str:
        """Return a one-line human summary of the streaming profile.

        Returns:
            A short, glanceable capability summary string.
        """
        if not self.any_streaming:
            return "batch only"
        bits: list[str] = []
        bits.append("mic" if self.streaming_input else "file-stream")
        if self.emits_partials:
            bits.append("partials")
        if self.re_segments:
            bits.append("corrections")
        if self.word_stability:
            bits.append("stable-prefix")
        if self.finality_mode:
            bits.append(self.finality_mode)
        return ", ".join(bits)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Discovered, instantiation-free metadata for one model key.

    Args:
        key: The model lookup key (``engine_id/model_name``).
        spec: The protocol's :class:`~standard_asr.ModelSpec`.
        properties: The engine's static :class:`~standard_asr.BaseProperties`, or
            ``None`` if the class could not be resolved without instantiation.
        capabilities: The engine's
            :class:`~standard_asr.DeclaredCapabilities`, or ``None`` as above.
        load_error: A description of why metadata could not be resolved, or
            ``None`` on success. A broken plugin is surfaced, never hidden.
    """

    key: str
    spec: ModelSpec
    properties: BaseProperties | None
    capabilities: DeclaredCapabilities | None
    load_error: str | None = None

    @property
    def description(self) -> str | None:
        """The engine's human-readable description, if declared.

        Returns:
            The properties' ``description`` field, or ``None``.
        """
        return self.properties.description if self.properties is not None else None

    def streaming_profile(self) -> StreamingProfile:
        """Summarize the model's streaming capabilities for display.

        Returns:
            A :class:`StreamingProfile`; an all-false profile when capabilities
            could not be resolved (fail-closed).
        """
        caps = self.capabilities
        if caps is None:
            return StreamingProfile(False, False, False, False, False, None)
        finality: str | None = None
        if caps.streaming is not None:
            finality = caps.streaming.finality_level.mode
        return StreamingProfile(
            streaming_input=caps.supports("streaming_input"),
            streaming_output=caps.supports("streaming_output"),
            emits_partials=caps.supports("streaming.emits_partials"),
            re_segments=caps.supports("streaming.re_segments"),
            word_stability=caps.supports("streaming.word_stability"),
            finality_mode=finality,
        )


def load_registry(*, strict: bool = False) -> ModelRegistry:
    """Discover installed Standard ASR engines.

    Args:
        strict: If ``True``, raise on an invalid entry point during discovery;
            otherwise invalid entries are warned about and skipped.

    Returns:
        The populated :class:`~standard_asr.ModelRegistry`.
    """
    return discover_models(strict=strict)


def describe_model(registry: ModelRegistry, key: str) -> ModelInfo:
    """Resolve instantiation-free metadata for one model key.

    Reads the engine's class-level ``properties`` / ``declared_capabilities`` via
    :meth:`ModelRegistry.engine_class`, which never constructs the engine. A
    plugin that fails to load is reported through ``load_error`` rather than
    raised, so one broken engine does not block listing the rest.

    Args:
        registry: A discovered registry.
        key: The model lookup key.

    Returns:
        A :class:`ModelInfo` for the key.
    """
    spec = registry.spec(key)
    try:
        cls = registry.engine_class(key)
        return ModelInfo(
            key=key,
            spec=spec,
            properties=cls.properties,
            capabilities=cls.declared_capabilities,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as load_error, not hidden
        return ModelInfo(
            key=key,
            spec=spec,
            properties=None,
            capabilities=None,
            load_error=f"{type(exc).__name__}: {exc}",
        )


def list_models(registry: ModelRegistry) -> list[ModelInfo]:
    """Resolve metadata for every discovered model key.

    Args:
        registry: A discovered registry.

    Returns:
        A list of :class:`ModelInfo`, one per model key, in registry order.
    """
    return [describe_model(registry, key) for key in registry.names()]


def config_schema(registry: ModelRegistry, key: str) -> dict[str, Any] | None:
    """Return a model's init-config JSON Schema without instantiation.

    Args:
        registry: A discovered registry.
        key: The model lookup key.

    Returns:
        The config JSON Schema (a pydantic ``model_json_schema()`` mapping), or
        ``None`` if the engine declares no ``config_type``.
    """
    return registry.config_schema(key)


__all__ = [
    "ModelInfo",
    "StreamingProfile",
    "config_schema",
    "describe_model",
    "list_models",
    "load_registry",
]
