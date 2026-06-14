# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Render an engine's init-config JSON Schema into a settings form.

The protocol publishes every engine's init config as a pydantic JSON Schema
(``registry.config_schema(key)``), with credential fields flagged via
``json_schema_extra`` (``"secret": true`` / ``"writeOnly": true`` /
``"format": "password"``). This module turns that schema into:

* a list of :class:`ConfigField` descriptors a UI can render, and
* a config dict built from CLI ``--set`` overrides and interactive prompts,

with two hard guarantees honouring the protocol's "security by default":

1. **Secrets are never echoed.** Secret fields are prompted with ``getpass`` and
   never printed, logged, or written to the events JSONL.
2. **Policy fields are not silently flipped.** ``strict`` and
   ``allow_private_urls`` are surfaced but never auto-prompted, mirroring the
   standard layer excluding them from environment fallback.

The built config is passed straight to ``registry.create(key, **config)``; the
standard layer validates it (a wrong type / missing required field raises the
protocol's own error, which the CLI surfaces verbatim).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

#: Config fields the form never auto-prompts for: the entry-point-derived engine
#: discriminator, and the two security/policy switches the standard layer
#: deliberately excludes from environment fallback (so they cannot be silently
#: relaxed). They remain settable explicitly via ``--set``.
_NON_INTERACTIVE_FIELDS = frozenset({"engine", "strict", "allow_private_urls"})


@dataclass(frozen=True, slots=True)
class ConfigField:
    """A single renderable init-config field.

    Args:
        name: The field name (the config keyword argument).
        type_label: A short human type label (``"string"``, ``"integer"``, ...).
        default: The schema default, or ``None`` if none.
        description: The field's description, if any.
        required: Whether the schema marks the field required.
        secret: Whether the field is credential-marked (never echo it).
        prompt_eligible: Whether the form should offer to prompt for it
            interactively (``False`` for the engine discriminator / policy
            switches).
        enum: Allowed values if the schema constrains them, else ``None``.
    """

    name: str
    type_label: str
    default: Any
    description: str | None
    required: bool
    secret: bool
    prompt_eligible: bool
    enum: list[Any] | None = None


def _type_label(prop: dict[str, Any]) -> str:
    """Derive a short human type label from a JSON Schema property node.

    Args:
        prop: A JSON Schema property sub-object.

    Returns:
        A short type label; ``"any"`` when the schema does not pin a type.
    """
    if "type" in prop:
        t = prop["type"]
        if t == "array":
            items = prop.get("items", {})
            inner = items.get("type", "any") if isinstance(items, dict) else "any"
            return f"array[{inner}]"
        return str(t)
    # anyOf (e.g. ``str | None``): pick the first non-null branch's type.
    for branch in prop.get("anyOf", []):
        if isinstance(branch, dict) and branch.get("type") not in (None, "null"):
            return str(branch["type"])
    return "any"


def _is_secret(prop: dict[str, Any]) -> bool:
    """Return whether a JSON Schema property is credential-marked.

    Recognizes the markers the standard layer's ``secret_field`` emits
    (``secret`` / ``writeOnly`` / ``format: password``), including when they sit
    inside an ``anyOf`` branch (``SecretStr | None``).

    Args:
        prop: A JSON Schema property sub-object.

    Returns:
        ``True`` if the field should be treated as a secret.
    """

    def marked(node: dict[str, Any]) -> bool:
        return bool(
            node.get("secret")
            or node.get("writeOnly")
            or node.get("format") == "password"
        )

    if marked(prop):
        return True
    return any(isinstance(b, dict) and marked(b) for b in prop.get("anyOf", []))


def fields_from_schema(schema: dict[str, Any] | None) -> list[ConfigField]:
    """Parse a config JSON Schema into renderable field descriptors.

    Args:
        schema: The engine's config JSON Schema, or ``None`` (no config type).

    Returns:
        The field descriptors in schema order; empty if ``schema`` is ``None``.
    """
    if not schema:
        return []
    required = set(schema.get("required", []))
    fields: list[ConfigField] = []
    for name, prop in schema.get("properties", {}).items():
        if not isinstance(prop, dict):
            continue
        secret = _is_secret(prop)
        fields.append(
            ConfigField(
                name=name,
                type_label=_type_label(prop),
                default=prop.get("default"),
                description=prop.get("description"),
                required=name in required,
                secret=secret,
                prompt_eligible=name not in _NON_INTERACTIVE_FIELDS,
                enum=prop.get("enum"),
            )
        )
    return fields


def coerce_value(field: ConfigField, raw: str) -> Any:
    """Coerce a raw ``--set`` / prompt string to the field's schema type.

    Conservative and forgiving: booleans and integers/numbers are parsed;
    arrays accept a JSON list or a comma-separated fallback; everything else is
    passed through as a string and left for the standard layer to validate (so a
    bad value still fails loudly at ``create`` with the protocol's own message).

    Args:
        field: The field descriptor (carries the target type label).
        raw: The raw user-supplied string.

    Returns:
        The coerced value.

    Raises:
        ValueError: If a boolean/number string cannot be parsed.
    """
    label = field.type_label
    if label == "boolean":
        lowered = raw.strip().lower()
        if lowered in ("true", "1", "yes", "y", "on"):
            return True
        if lowered in ("false", "0", "no", "n", "off"):
            return False
        raise ValueError(f"{field.name}: expected a boolean, got {raw!r}.")
    if label == "integer":
        return int(raw)
    if label == "number":
        return float(raw)
    if label.startswith("array"):
        text = raw.strip()
        if text.startswith("["):
            return json.loads(text)
        return [item.strip() for item in text.split(",") if item.strip()]
    return raw


def parse_overrides(
    fields: list[ConfigField], overrides: list[str]
) -> dict[str, Any]:
    """Parse ``--set KEY=VALUE`` overrides into a coerced config dict.

    Args:
        fields: The known field descriptors (for type coercion).
        overrides: Raw ``"key=value"`` strings (repeatable CLI flag).

    Returns:
        A mapping of field name to coerced value.

    Raises:
        ValueError: On a malformed override (no ``=``), an unknown field, or a
            value that cannot be coerced.
    """
    by_name = {f.name: f for f in fields}
    out: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}.")
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in by_name:
            known = ", ".join(sorted(by_name)) or "<none>"
            raise ValueError(f"Unknown config field {key!r}. Known fields: {known}.")
        out[key] = coerce_value(by_name[key], value)
    return out


def redacted_config(fields: list[ConfigField], config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``config`` safe to display/log (secrets masked).

    Args:
        fields: The field descriptors (to know which keys are secret).
        config: The built config mapping.

    Returns:
        A copy with every secret field's value replaced by ``"********"``.
    """
    secret_names = {f.name for f in fields if f.secret}
    return {
        key: ("********" if key in secret_names and value is not None else value)
        for key, value in config.items()
    }


__all__ = [
    "ConfigField",
    "coerce_value",
    "fields_from_schema",
    "parse_overrides",
    "redacted_config",
]
