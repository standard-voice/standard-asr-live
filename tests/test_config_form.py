# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for JSON Schema -> settings form parsing and secret handling."""

from __future__ import annotations

import pytest
from standard_asr import discover_models

from standard_asr_live.config_form import (
    coerce_value,
    fields_from_schema,
    parse_overrides,
    redacted_config,
)
from standard_asr_live.errors import LiveAppError

# A schema fragment mirroring the standard layer's secret_field markers.
_SCHEMA = {
    "properties": {
        "engine": {"type": "string", "default": "demo"},
        "strict": {"type": "boolean", "default": True, "description": "policy"},
        "api_key": {
            "anyOf": [
                {"type": "string", "format": "password", "writeOnly": True, "secret": True},
                {"type": "null"},
            ],
            "description": "Secret API key / token.",
        },
        "beam_size": {"type": "integer", "default": 5, "description": "beam"},
        "tags": {"type": "array", "items": {"type": "string"}, "description": "labels"},
    },
    "required": ["engine"],
}


def test_fields_parsed_with_types_and_flags() -> None:
    """Fields carry type label, default, required and secret flags."""
    fields = {f.name: f for f in fields_from_schema(_SCHEMA)}
    assert fields["beam_size"].type_label == "integer"
    assert fields["beam_size"].default == 5
    assert fields["tags"].type_label == "array[string]"
    assert fields["engine"].required is True


def test_secret_field_detected_inside_anyof() -> None:
    """A SecretStr | None secret marker is detected inside an anyOf branch."""
    fields = {f.name: f for f in fields_from_schema(_SCHEMA)}
    assert fields["api_key"].secret is True
    # The engine discriminator and policy switches are not interactively prompted.
    assert fields["engine"].prompt_eligible is False
    assert fields["strict"].prompt_eligible is False
    assert fields["beam_size"].prompt_eligible is True


def test_none_schema_yields_no_fields() -> None:
    """A None schema (engine declares no config_type) yields no fields."""
    assert fields_from_schema(None) == []


def test_coercion_by_type() -> None:
    """Override strings are coerced to the field's schema type."""
    fields = {f.name: f for f in fields_from_schema(_SCHEMA)}
    assert coerce_value(fields["beam_size"], "8") == 8
    assert coerce_value(fields["strict"], "false") is False
    assert coerce_value(fields["tags"], "a, b, c") == ["a", "b", "c"]
    assert coerce_value(fields["tags"], '["x", "y"]') == ["x", "y"]


def test_coercion_rejects_bad_boolean() -> None:
    """A non-boolean string for a boolean field fails loudly."""
    fields = {f.name: f for f in fields_from_schema(_SCHEMA)}
    with pytest.raises(ValueError):
        coerce_value(fields["strict"], "maybe")


def test_parse_overrides_unknown_field_errors() -> None:
    """An unknown --set key fails with a helpful message."""
    fields = fields_from_schema(_SCHEMA)
    with pytest.raises(LiveAppError, match="Unknown config field"):
        parse_overrides(fields, ["nope=1"])


def test_parse_overrides_requires_equals() -> None:
    """A --set without '=' fails loudly."""
    fields = fields_from_schema(_SCHEMA)
    with pytest.raises(LiveAppError, match="KEY=VALUE"):
        parse_overrides(fields, ["justakey"])


def test_parse_overrides_bad_value_is_wrapped() -> None:
    """A value that fails coercion is surfaced as a clean --set error, not a ValueError."""
    fields = fields_from_schema(_SCHEMA)
    with pytest.raises(LiveAppError, match="--set strict"):
        parse_overrides(fields, ["strict=maybe"])


def test_redacted_config_masks_secret_values() -> None:
    """Secret values are masked in the displayable config, others kept."""
    fields = fields_from_schema(_SCHEMA)
    config = {"api_key": "sk-supersecret", "beam_size": 8}
    red = redacted_config(fields, config)
    assert red["api_key"] == "********"
    assert red["beam_size"] == 8
    assert "sk-supersecret" not in str(red)


def test_real_engine_schema_has_no_unexpected_secret() -> None:
    """The dummy engine's real schema parses and marks no field secret."""
    registry = discover_models()
    if "dummy/echo" not in registry.names():
        pytest.skip("dummy engine not installed")
    fields = fields_from_schema(registry.config_schema("dummy/echo"))
    names = {f.name for f in fields}
    assert "message" in names
    assert not any(f.secret for f in fields)  # dummy has no credentials
