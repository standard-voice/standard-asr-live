# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for standard-asr-live.

Subcommands:

* ``models``  -- list discovered engines and their streaming capabilities.
* ``show``    -- full properties + capability tree + config schema for a model.
* ``run``     -- the live transcription app (the centerpiece).
* ``doctor``  -- environment / dependency diagnostics (via ``standard_asr.diagnose``).

All engine interaction goes through the protocol; no concrete engine is imported.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.json import JSON
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .audio_io import ChunkPlan, list_input_devices
from .config_form import (
    ConfigField,
    fields_from_schema,
    parse_overrides,
    redacted_config,
)
from .discovery import ModelInfo, describe_model, list_models, load_registry
from .driver import DriveConfig, Mode, Source, drive, select_mode
from .engine_view import LiveTranscript
from .errors import LiveAppError, NoEngineSelectedError, describe_exception
from .export import export_result
from .view import (
    render_banner,
    render_diagnostics,
    render_status,
    render_transcript,
)

if TYPE_CHECKING:
    from standard_asr import StandardASR

#: Default demo audio shipped with the Standard ASR repo (≈57 s English).
_DEFAULT_DEMO_AUDIO = (
    "/Users/tim/LocalData/coding/LongTermProjects/4-Standard-Voice/standard_asr/"
    "reference/standard_asr_test_audio_english.m4a"
)


# --------------------------------------------------------------------------- #
# models / show
# --------------------------------------------------------------------------- #
def _cmd_models(args: argparse.Namespace, console: Console) -> int:
    """List discovered models with their streaming capability summary.

    Args:
        args: Parsed CLI arguments.
        console: Output console.

    Returns:
        Process exit code.
    """
    registry = load_registry(strict=args.strict)
    infos = list_models(registry)
    if not infos:
        console.print(
            Panel(
                Text(
                    "No Standard ASR engines are installed.\n\n"
                    "Install one, e.g. the cookbook faster-whisper plugin:\n"
                    "  uv pip install -e <standard_asr>/cookbook/std_faster_whisper\n"
                    "or the dummy demo engine:\n"
                    "  uv pip install -e <standard_asr>/cookbook/std_dummy_asr",
                    style="yellow",
                ),
                title="no engines found",
                border_style="yellow",
            )
        )
        return 1
    table = Table(title="Discovered Standard ASR engines", title_style="bold")
    table.add_column("model key", style="cyan", no_wrap=True)
    table.add_column("streaming", style="green")
    table.add_column("description", style="dim")
    for info in infos:
        if info.load_error is not None:
            table.add_row(info.key, Text("load error", style="red"), info.load_error)
            continue
        profile = info.streaming_profile()
        table.add_row(info.key, profile.headline(), info.description or "")
    console.print(table)
    if registry.shadowed_engine_ids:
        console.print(
            Text(
                f"note: ambiguous engine ids (multiple distributions): "
                f"{sorted(registry.shadowed_engine_ids)}",
                style="yellow",
            )
        )
    return 0


def _cmd_show(args: argparse.Namespace, console: Console) -> int:
    """Show full properties, capabilities, and config schema for one model.

    Args:
        args: Parsed CLI arguments.
        console: Output console.

    Returns:
        Process exit code.
    """
    registry = load_registry(strict=args.strict)
    if args.model not in registry.names():
        console.print(
            Text(f"Model {args.model!r} not found. Known: {registry.names()}", style="red")
        )
        return 1
    info = describe_model(registry, args.model)
    _print_model_detail(info, registry, console)
    return 0


def _print_model_detail(info: ModelInfo, registry, console: Console) -> None:
    """Print the detailed model view (properties + capabilities + config).

    Args:
        info: The resolved model metadata.
        registry: The discovered registry (for the config schema).
        console: Output console.

    Returns:
        None.
    """
    console.print(Panel(Text(info.key, style="bold cyan"), border_style="cyan"))
    if info.load_error is not None:
        console.print(Text(f"could not load engine class: {info.load_error}", style="red"))
        return

    props = info.properties
    if props is not None:
        prop_table = Table(title="Properties", title_style="bold", show_header=False)
        prop_table.add_column(style="dim", justify="right")
        prop_table.add_column()
        prop_table.add_row("engine_id", props.engine_id)
        prop_table.add_row("model_name", props.model_name or "(default)")
        prop_table.add_row("protocol_version", props.protocol_version)
        accepted = ", ".join(sorted(k.value for k in props.accepted_input))
        prop_table.add_row("accepted_input", accepted)
        prop_table.add_row("native_sample_rate", str(props.native_sample_rate))
        prop_table.add_row("accepted_sample_rates", str(props.accepted_sample_rates))
        prop_table.add_row("selectable_languages", ", ".join(props.selectable_languages) or "—")
        prop_table.add_row("wire_encodings", str(props.wire_encodings))
        console.print(prop_table)

    caps = info.capabilities
    if caps is not None:
        console.print(Text("Capabilities (canonical JSON):", style="bold"))
        console.print(JSON.from_data(caps.canonical_json()))

    schema = registry.config_schema(info.key)
    fields = fields_from_schema(schema)
    if fields:
        cfg_table = Table(title="Init config", title_style="bold")
        cfg_table.add_column("field", style="cyan")
        cfg_table.add_column("type", style="dim")
        cfg_table.add_column("default")
        cfg_table.add_column("secret", justify="center")
        cfg_table.add_column("description", style="dim")
        for field in fields:
            cfg_table.add_row(
                field.name,
                field.type_label,
                "—" if field.default is None else str(field.default),
                "yes" if field.secret else "",
                field.description or "",
            )
        console.print(cfg_table)


# --------------------------------------------------------------------------- #
# config building
# --------------------------------------------------------------------------- #
def _build_config(
    fields: list[ConfigField],
    overrides: list[str],
    *,
    interactive: bool,
    console: Console,
) -> dict:
    """Assemble the engine config from --set overrides and optional prompts.

    Secret fields are prompted with ``getpass`` and never echoed. Non-interactive
    runs use only ``--set`` overrides plus schema defaults (filled by the engine).

    Args:
        fields: The config field descriptors.
        overrides: Raw ``key=value`` override strings.
        interactive: Whether to prompt for prompt-eligible fields.
        console: Output console.

    Returns:
        The assembled config mapping (only explicitly-set keys; the standard
        layer applies defaults for the rest).
    """
    config = parse_overrides(fields, overrides)
    if interactive:
        for field in fields:
            if field.name in config or not field.prompt_eligible:
                continue
            if field.secret:
                # Prompt only for secrets that are required, to avoid nagging.
                if not field.required:
                    continue
                value = getpass.getpass(f"  {field.name} (secret, hidden): ")
                if value:
                    config[field.name] = value
            # Non-secret optional fields keep their schema default unless --set.
    return config


def _create_engine(
    registry, key: str, config: dict, console: Console
) -> StandardASR:
    """Create an engine instance, surfacing protocol errors verbatim.

    Args:
        registry: The discovered registry.
        key: The model key.
        config: The assembled config mapping.
        console: Output console.

    Returns:
        The constructed engine.

    Raises:
        LiveAppError: Wrapping any construction failure with a clear message
            (the underlying protocol error message is preserved; secrets are
            never echoed because the standard layer does not embed them).
    """
    try:
        return registry.create(key, **config)
    except Exception as exc:  # noqa: BLE001 - re-raised as a friendly app error
        raise LiveAppError(
            f"Could not create engine {key!r}: {describe_exception(exc)}"
        ) from exc


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def _select_model_key(args: argparse.Namespace, registry, console: Console) -> str:
    """Resolve the model key from args or an interactive picker.

    Args:
        args: Parsed CLI arguments.
        registry: The discovered registry.
        console: Output console.

    Returns:
        The chosen model key.

    Raises:
        NoEngineSelectedError: If no model could be chosen.
    """
    names = registry.names()
    if not names:
        raise NoEngineSelectedError(
            "No Standard ASR engines are installed. Install a plugin first "
            "(see 'standard-asr-live models')."
        )
    if args.model:
        if args.model not in names:
            raise NoEngineSelectedError(
                f"Model {args.model!r} not found. Known models: {names}."
            )
        return args.model
    if not sys.stdin.isatty():
        # Non-interactive with no model: pick the first deterministically.
        return names[0]
    console.print(Text("Available models:", style="bold"))
    for i, name in enumerate(names):
        console.print(f"  [{i}] {name}")
    raw = console.input("Select a model [0]: ").strip() or "0"
    try:
        return names[int(raw)]
    except (ValueError, IndexError) as exc:
        raise NoEngineSelectedError(f"Invalid selection {raw!r}.") from exc


def _resolve_sample_rate(engine: StandardASR) -> int:
    """Choose a PCM wire sample rate the engine will accept.

    Prefers the engine's ``native_sample_rate`` (which the reachability
    invariants guarantee is acceptable); falls back to 16 kHz.

    Args:
        engine: The engine instance.

    Returns:
        A sample rate in Hz suitable for the wire ``AudioFormat``.
    """
    props = engine.properties
    native = getattr(props, "native_sample_rate", 16000)
    required = getattr(props, "required_input_sample_rate", None)
    return int(required or native or 16000)


def _runtime_params(args: argparse.Namespace):
    """Build RuntimeParams from CLI args (language only, for portability).

    Args:
        args: Parsed CLI arguments.

    Returns:
        A :class:`~standard_asr.RuntimeParams`.
    """
    from standard_asr import RuntimeParams

    return RuntimeParams(language=args.language) if args.language else RuntimeParams()


def _cmd_run(args: argparse.Namespace, console: Console) -> int:
    """Run the live transcription app.

    Args:
        args: Parsed CLI arguments.
        console: Output console.

    Returns:
        Process exit code.
    """
    if args.list_devices:
        return _list_devices(console)

    registry = load_registry()
    key = _select_model_key(args, registry, console)
    info = describe_model(registry, key)
    if info.load_error is not None:
        raise LiveAppError(f"Engine {key!r} failed to load: {info.load_error}")

    fields = fields_from_schema(registry.config_schema(key))
    interactive = sys.stdin.isatty() and not args.no_prompt
    config = _build_config(fields, args.set, interactive=interactive, console=console)
    console.print(
        Text(f"config: {json.dumps(redacted_config(fields, config))}", style="dim")
    )
    engine = _create_engine(registry, key, config, console)

    source, file_path = _resolve_source(args)
    sample_rate = _resolve_sample_rate(engine)
    plan = ChunkPlan(
        sample_rate=sample_rate,
        chunk_ms=args.chunk_ms,
        speed=args.speed,
        paced=not args.no_pace,
    )
    cfg = DriveConfig(
        source=source,
        file_path=file_path,
        plan=plan,
        params=_runtime_params(args),
        device=args.device,
        use_sync_bridge=args.sync,
        strict_lifecycle=args.strict_lifecycle,
    )
    mode = select_mode(engine)
    console.print(
        Text(
            f"Running {key} | mode={mode.value} | source={source.value} | "
            f"rate={sample_rate}Hz | {'sync-bridge' if args.sync else 'async'}",
            style="bold",
        )
    )
    return _run_live(engine, cfg, key, mode, args, console)


def _resolve_source(args: argparse.Namespace) -> tuple[Source, str | None]:
    """Resolve the audio source (mic or file) from CLI args.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A ``(source, file_path)`` pair; ``file_path`` is ``None`` for mic.

    Raises:
        LiveAppError: If a given file path does not exist.
    """
    if args.mic:
        return Source.MIC, None
    path = args.file or _DEFAULT_DEMO_AUDIO
    if not Path(path).is_file():
        raise LiveAppError(
            f"Audio file not found: {path!r}. Pass --file PATH or --mic."
        )
    return Source.FILE, path


def _list_devices(console: Console) -> int:
    """Print available microphone input devices.

    Args:
        console: Output console.

    Returns:
        Process exit code.
    """
    devices = list_input_devices()
    if not devices:
        console.print(Text("No input devices found.", style="yellow"))
        return 1
    table = Table(title="Microphone input devices")
    table.add_column("index", justify="right", style="cyan")
    table.add_column("name")
    table.add_column("channels", justify="right", style="dim")
    for index, name, channels in devices:
        table.add_row(str(index), name, str(channels))
    console.print(table)
    return 0


def _run_live(
    engine: StandardASR,
    cfg: DriveConfig,
    key: str,
    mode: Mode,
    args: argparse.Namespace,
    console: Console,
) -> int:
    """Drive a session and render the live transcript until it ends.

    Args:
        engine: The engine instance.
        cfg: The drive configuration.
        key: The model key (for the header).
        mode: The selected transcription mode.
        args: Parsed CLI arguments.
        console: Output console.

    Returns:
        Process exit code (0 on clean ``done``, 2 on a terminal error).
    """
    state = LiveTranscript()
    started = time.monotonic()
    session = drive(engine, cfg)
    events_log = _open_events_log(args)
    spinner_note = (
        Text("batch mode: transcribing whole file (no live partials)...", style="yellow")
        if mode is Mode.BATCH
        else None
    )

    def _render():
        elapsed = time.monotonic() - started
        parts = []
        if spinner_note is not None and not state.is_finished():
            parts.append(Panel(spinner_note, border_style="yellow"))
        banner = render_banner(state)
        if banner is not None:
            parts.append(banner)
        parts.append(render_transcript(state))
        bottom = Table.grid(expand=True)
        bottom.add_column(ratio=1)
        bottom.add_column(ratio=1)
        bottom.add_row(
            render_status(state, elapsed=elapsed, mode=mode.value, engine_key=key),
            render_diagnostics(session.diagnostics()),
        )
        parts.append(bottom)
        from rich.console import Group

        return Group(*parts)

    use_live = sys.stdout.isatty() and not args.plain
    try:
        if use_live:
            with Live(_render(), console=console, refresh_per_second=12, screen=False) as live:
                for event in session.events:
                    state.apply(event)
                    _log_event(events_log, event)
                    live.update(_render())
                live.update(_render())
        else:
            # Non-TTY / --plain: fold all events, then print the final frame once.
            for event in session.events:
                state.apply(event)
                _log_event(events_log, event)
            console.print(_render())
    finally:
        if events_log is not None:
            events_log.close()

    return _finish(state, session, args, console)


def _finish(
    state: LiveTranscript, session, args: argparse.Namespace, console: Console
) -> int:
    """Print the final result, diagnostics, and perform export if requested.

    Args:
        state: The final reducer state.
        session: The drive session (for the authoritative result).
        args: Parsed CLI arguments.
        console: Output console.

    Returns:
        Process exit code (0 clean, 2 if the stream ended in a terminal error).
    """
    result = session.result()
    console.print()
    if state.ended_in_error and state.terminal is not None:
        term = state.terminal
        detail = term.extra.get("detail", "") if term.extra else ""
        console.print(
            Panel(
                Text(f"stream ended with error: {term.code} {detail}", style="red"),
                border_style="red",
            )
        )

    if result is not None:
        summary = Table.grid(padding=(0, 1))
        summary.add_column(style="dim", justify="right")
        summary.add_column()
        summary.add_row("transcript", result.text or "(empty)")
        summary.add_row("language", result.detected_language or "—")
        summary.add_row(
            "duration", f"{result.duration:.2f}s" if result.duration is not None else "—"
        )
        summary.add_row("segments", str(len(result.segments or [])))
        console.print(Panel(summary, title="FINAL RESULT", border_style="blue", title_align="left"))

        if args.export:
            exported = export_result(result, args.export)
            console.print(
                Text(
                    "exported: " + ", ".join(str(p) for p in exported.as_list()),
                    style="green",
                )
            )
    else:
        console.print(Text("No result was produced.", style="yellow"))

    return 2 if state.ended_in_error else 0


def _open_events_log(args: argparse.Namespace):
    """Open the JSONL events log file if --json-events was given.

    Args:
        args: Parsed CLI arguments.

    Returns:
        An open text file handle, or ``None``.
    """
    if not args.json_events:
        return None
    return open(args.json_events, "w", encoding="utf-8")


def _log_event(handle, event) -> None:
    """Append one event to the JSONL log, if logging is enabled.

    Args:
        handle: The open log file, or ``None``.
        event: The transcription event to record.

    Returns:
        None.
    """
    if handle is None:
        return
    # model_dump(exclude_none) keeps the log compact; no secrets ever flow here.
    handle.write(json.dumps(event.model_dump(exclude_none=True), default=str) + "\n")


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
def _cmd_doctor(args: argparse.Namespace, console: Console) -> int:
    """Run environment / dependency diagnostics via the protocol's doctor.

    Args:
        args: Parsed CLI arguments.
        console: Output console.

    Returns:
        Process exit code.
    """
    import shutil

    from standard_asr import diagnose

    table = Table(title="Environment", title_style="bold", show_header=False)
    table.add_column(style="dim", justify="right")
    table.add_column()
    table.add_row("standard-asr-live", __version__)
    table.add_row("python", sys.version.split()[0])
    table.add_row("ffmpeg", shutil.which("ffmpeg") or "[red]not found[/red]")
    try:
        import sounddevice  # noqa: F401

        table.add_row("sounddevice", "available")
    except Exception as exc:  # noqa: BLE001
        table.add_row("sounddevice", f"[yellow]unavailable: {describe_exception(exc)}[/yellow]")
    console.print(table)

    registry = load_registry()
    console.print(Text(f"discovered engines: {registry.names() or '(none)'}", style="cyan"))

    # diagnose() is keyword-only (group=...) and returns a DoctorReport; it checks
    # numpy compatibility across installed plugins (a real "dependency hell" guard).
    report = diagnose()
    dep_table = Table(title="standard_asr.doctor (dependency conflicts)", show_header=False)
    dep_table.add_column(style="dim", justify="right")
    dep_table.add_column()
    dep_table.add_row("python", report.python_version)
    dep_table.add_row("clean", "yes" if report.is_clean else "[red]no[/red]")
    dep_table.add_row("conflicts", str(report.conflicts) if report.has_conflict else "none")
    for note in report.notes:
        dep_table.add_row("note", note)
    console.print(dep_table)
    return 0


# --------------------------------------------------------------------------- #
# argument parsing / entry point
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="standard-asr-live",
        description=(
            "Live streaming speech-to-text in the terminal, against any Standard ASR engine."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_models = sub.add_parser("models", help="List discovered engines and streaming capabilities.")
    p_models.add_argument("--strict", action="store_true", help="Fail on invalid entry points.")
    p_models.set_defaults(func=_cmd_models)

    p_show = sub.add_parser("show", help="Show properties, capabilities, and config schema.")
    p_show.add_argument("model", help="Model key, e.g. 'faster-whisper/large-v3'.")
    p_show.add_argument("--strict", action="store_true", help="Fail on invalid entry points.")
    p_show.set_defaults(func=_cmd_show)

    p_run = sub.add_parser("run", help="Run the live transcription app.")
    p_run.add_argument("model", nargs="?", default=None, help="Model key (interactive if omitted).")
    src = p_run.add_mutually_exclusive_group()
    src.add_argument("--file", help="Audio file to stream (default: bundled demo audio).")
    src.add_argument("--mic", action="store_true", help="Capture from the microphone.")
    p_run.add_argument("--list-devices", action="store_true", help="List input devices and exit.")
    p_run.add_argument("--device", type=int, default=None, help="Microphone device index.")
    p_run.add_argument("--language", default=None, help="BCP-47 language tag or 'auto'.")
    p_run.add_argument("--chunk-ms", type=int, default=100, help="Wire chunk size in ms.")
    p_run.add_argument("--speed", type=float, default=1.0, help="File pace multiplier (1.0=live).")
    p_run.add_argument("--no-pace", action="store_true", help="Feed the file as fast as possible.")
    p_run.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="Config override (repeats)."
    )
    p_run.add_argument("--export", default=None, metavar="DIR", help="Export TXT/SRT/VTT to DIR.")
    p_run.add_argument("--sync", action="store_true", help="Drive via the SyncSession bridge.")
    p_run.add_argument(
        "--strict-lifecycle", action="store_true", help="Raise on illegal lifecycle transitions."
    )
    p_run.add_argument("--json-events", default=None, metavar="PATH", help="Log events to JSONL.")
    p_run.add_argument("--no-prompt", action="store_true", help="Never prompt; use defaults/--set.")
    p_run.add_argument("--plain", action="store_true", help="Plain output (no live redraw).")
    p_run.set_defaults(func=_cmd_run)

    p_doctor = sub.add_parser("doctor", help="Environment / dependency diagnostics.")
    p_doctor.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    console = Console(no_color=getattr(args, "plain", False))
    try:
        return int(args.func(args, console))
    except LiveAppError as exc:
        console.print(Text(f"error: {exc}", style="bold red"))
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive interrupt
        console.print(Text("\ninterrupted.", style="yellow"))
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
