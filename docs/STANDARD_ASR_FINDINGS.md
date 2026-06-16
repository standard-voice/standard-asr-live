# Standard ASR — application-developer findings

Findings from building **standard-asr-live** (a live streaming captioning CLI) as
a demanding consumer of the `standard-asr` v0.1.0 protocol on the
`main` branch. The app developer is the protocol's primary
stakeholder, so this is written to be directly actionable.

Each finding: **what happened**, **why it matters**, **suggested improvement**.
Verdicts are empirical (re-verified against the installed library). Overall the
protocol is *remarkably* solid and well-documented — most of what follows is
papercuts and missing conveniences, not design holes. The big wins are called out
in §A at the end.

Severity legend: **[blocker]** stops you, **[friction]** costs real time,
**[papercut]** minor annoyance, **[doc]** documentation gap.

---

## 1. [friction] No canonical wire-encoding constant; apps hardcode `"pcm_s16le"`

**What happened.** To open an incremental streaming session you build
`AudioFormat(encoding="pcm_s16le", sample_rate=sr, channels=1)`. The spec calls
`pcm_s16le` "the canonical wire encoding," but it is **not exported as a named
constant** anywhere in the public API (`dir(standard_asr)` has no
`CANONICAL_WIRE_ENCODING` / `DEFAULT_WIRE_ENCODING`; `AudioFormat.encoding` is a
free-form `str` with no enum/const in its JSON Schema). Every app and every
engine that produces PCM must hardcode the magic string `"pcm_s16le"` and hope it
matches what engines declare in `wire_encodings`.

**Why it matters.** Magic strings drift. A typo (`"pcm_16le"`) is a silent
runtime mismatch, not a type error. Cross-language wire clients (a stated G.5
goal) especially benefit from one authoritative spelling.

**Suggested improvement.** Export a constant, e.g.
`standard_asr.CANONICAL_WIRE_ENCODING = "pcm_s16le"`, and ideally a small
`Literal`/enum of known wire encodings (`pcm_s16le`, `mulaw`, …) that
`AudioFormat.encoding` and `BaseProperties.wire_encodings` can reference. Mention
it in `discover_and_use.md` §7.

---

## 2. [friction] No helper to build the wire `AudioFormat` an engine wants — yet the library already has one (private)

**What happened.** Given an engine, an app that wants to stream from a mic/file
must figure out the right `AudioFormat` itself: read `properties.wire_encodings`
(pick one), `properties.native_sample_rate` (or `required_input_sample_rate`),
and assemble `AudioFormat(encoding=…, sample_rate=…, channels=1)`. There is no
public helper (`dir(engine)` / `dir(engine.properties)` / the `standard_asr`
namespace have no `preferred_wire_format` / `default_audio_format` /
`wire_format_for`). **The library already contains exactly this logic, twice, but
both copies are private and unexported:** `cli.py::_streaming_audio_format(engine)`
and `compliance.py::_synthesize_probe_audio_format(engine)`. My app reimplemented
it (`_streaming_audio_format` in `cli.py`, `_resolve_sample_rate`), i.e. I wrote
code the library already has.

**Why it matters.** This is *the* first thing every streaming app must do, it's
fiddly (which encoding? native vs required rate? what if `wire_encodings` is
`None`?), and getting it wrong yields a fail-closed `UnsupportedFeatureError` at
session open. The protocol owns the negotiation rules — it should own this.

**Suggested improvement.** Promote the private helper to public, e.g.
`BaseProperties.preferred_wire_format() -> AudioFormat | None` (or
`EngineBase.recommended_audio_format()`), returning the engine's canonical
incremental wire format, or `None` when the engine self-manages its wire format
(no `wire_encodings`). It already exists internally; exporting it removes a
whole class of "did I pick the right rate/encoding" app bugs.

---

## 3. [friction] `supports("streaming")` returns `True` for any engine declaring the streaming domain — a footgun for capability gating

**What happened.** To decide "can this engine do live streaming," the natural
first guess is `engine.supports("streaming")`. It returns `True` whenever the
`streaming` *mode-domain* merely **exists** — even for an engine that only
declares `streaming_output` and cannot accept incremental mic input. The actual
live-input gate is the **top-level** `supports("streaming_input")`, which is a
*sibling* of the `streaming` domain, not a child. So:

| dot-path | returns | reality |
|---|---|---|
| `supports("streaming")` | `True` | only means "a streaming domain is declared" |
| `supports("streaming_input")` | `True`/`False` | the real "can feed live audio" gate (top-level) |
| `supports("streaming.streaming_input")` | `False` | wrong path (it's top-level, not nested) |
| `supports("emits_partials")` | `False` | wrong path (it IS nested under `streaming.`) |
| `supports("streaming.emits_partials")` | `True`/`False` | correct |

The split — `streaming_input`/`streaming_output`/`self_resamples` at top level,
but `emits_partials`/`re_segments`/`word_stability`/… under `streaming.*` — is
documented and principled (orthogonal axes vs mode-domain caps), but it is **very
easy to query the wrong path**, and the wrong path fails *closed* (returns
`False`) — silently, exactly the kind of quiet wrong answer the project warns
against.

**Why it matters.** A capability query that returns a confidently-wrong boolean
(`supports("streaming")` → True for a batch-mic-incapable engine; or a mistyped
nested path → False) leads to either a confusing late error or a silently skipped
feature. My driver had to learn the exact paths by reading source + the explore
agent.

**Suggested improvement.** Two low-cost options: (a) add convenience predicates
like `engine.can_stream_incrementally()` / `can_stream_output()` that encode the
"top-level `streaming_input` AND streaming domain present" intent; and/or (b) make
`supports()` raise (or warn) on a **syntactically valid but unknown** dot-path
(e.g. `streaming.streaming_input`, `emits_partials`) instead of returning `False`
— a typo'd capability path is a code bug, and fail-closed-silent hides it. At
minimum, document the top-level-vs-nested split prominently in
`discover_and_use.md` §5 with the exact paths.

---

## 4. [papercut] `StreamReducer.result()` double-spaces segments whose text has trailing whitespace

**What happened.** `StreamReducer` joins committed `final` segments with
`" ".join(...)` without trimming each segment's own edge whitespace. A segment
whose text is `"the quick "` (legitimately frozen with a trailing space) followed
by `"brown fox"` yields `result().text == "the quick  brown fox"` (two spaces).
Verified directly.

**Why it matters.** It's a visible cosmetic defect in the authoritative final
transcript / export. It surfaced in my real run: a `closed`/`final` whose text
ended in a space produced a double space in the exported `.txt`. Apps can't easily
pre-fix it because the reducer is internal to the session.

**Suggested improvement.** In `StreamReducer.result()`, join with
`" ".join(s.text.strip() for s in segments)` (or collapse runs of whitespace) —
matching what a human reads. If trailing spaces are ever semantically meaningful
they should not survive into the joined top-level `text` anyway.

---

## 5. [doc/friction] Catch `pydantic.ValidationError`, not `ConfigError`, for bad init config

**What happened.** `registry.create(key, **config)` with a wrong-typed field
(`cpu_threads="not-an-int"`) raises a clean `pydantic.ValidationError` — good
message — but it is **not** a subclass of `standard_asr.ConfigError`
(`ConfigError` MRO is `ConfigError → StandardASRError → ValueError`). An app that
defensively does `except ConfigError` to present config-form errors will **miss**
type-validation failures and let them crash. (They do share `ValueError`, so
`except ValueError` works — but that's a wide net.)

**Why it matters.** "Render a config form, build the engine, show errors nicely"
is the canonical config flow (G.3.1). The error taxonomy is subtle here: some
config problems raise `ConfigError` (e.g. bad `default_language` via
`_validate_language_config`), others raise raw pydantic `ValidationError`. An app
must catch both, which isn't obvious.

**Suggested improvement.** Either (a) have `registry.create` wrap pydantic
validation failures in `ConfigError` (preserving `__cause__`) so there's **one**
config-error type to catch, or (b) document explicitly in `create`'s docstring
and `discover_and_use.md` that callers must catch `pydantic.ValidationError`
**and** `ConfigError`. The docstring currently says "factory errors propagate
unchanged" but doesn't name the pydantic case as the common one.

---

## 6. [papercut] Discovery prints warnings to stderr/logging with no programmatic surface for "default-model" entries

**What happened.** `discover_models()` emits a logging warning for every
empty-model-name entry point (the `engine/` default-model convention the cookbook
itself uses: `"dummy/"`). With `logging.basicConfig()` (which any CLI sets up)
this prints:

```
model_name is empty for a standard_asr.models entry point. Empty names are
allowed but discouraged; document the default clearly.
```

So a perfectly valid, documented pattern (a default-model alias) produces a
scary-looking warning on **every** discovery, including in my app's normal
startup. There's no structured way to know "these keys are default aliases" other
than the empty `model_name` on the `ModelSpec`.

**Why it matters.** Warnings that fire on the happy path train users to ignore
warnings. The cookbook ships `dummy/` *and* `dummy/echo`, so the demo's own
output is noisy.

**Suggested improvement.** Downgrade the empty-name notice to `INFO` (or drop it
when an explicit non-empty sibling exists), since the default-model alias is an
endorsed pattern. Keep the `strict=True` rejection for genuinely malformed names.

---

## 7. [papercut] Engine-identity collision warning is good, but the only resolution is "uninstall one"

**What happened.** When two installed distributions declare the same `engine_id`
(I hit this deliberately: a `faster-whisper/tiny` preset package alongside the
cookbook `std-faster-whisper`), discovery warns:

```
Engine-identity collision: engine_id 'faster-whisper' is provided by multiple
distributions (...). config.engine routing is ambiguous; install only one
provider for this engine_id, or have authors choose distinct engine_ids.
```

This is **correct and valuable** (it surfaces a real ambiguity, exposed via
`registry.shadowed_engine_ids`). The papercut: adding a *new preset* for an
existing engine is a legitimate thing an engine author / power user wants to do
(faster-whisper has ~15 sizes; the cookbook registers 3), but the protocol treats
*any* second provider of the `engine_id` as a collision — even when both resolve
to the same engine class and only add new model keys.

**Why it matters.** It nudges toward forking the whole plugin to add one preset.
The model key (`engine/model`) is already unique; the collision is at the
`engine_id` (config routing) level.

**Suggested improvement.** Consider distinguishing "same `engine_id`, disjoint
model keys, compatible config type" (a benign preset extension) from "same
`engine_id`, conflicting config routing" (the real ambiguity). At least document
the recommended way to publish *additional presets* for an existing engine
without tripping the collision.

---

## 8. [papercut] `diagnose()` is keyword-only with no registry argument and no `to_dict()`

**What happened.** I wired a `doctor` command and first wrote `diagnose(registry)`
(reasonable guess — it's the dependency doctor). It raised `TypeError: diagnose()
takes 0 positional arguments`. The real signature is `diagnose(*, group=...)` and
it returns a `DoctorReport` with attributes (`conflicts`, `has_conflict`,
`is_clean`, `notes`, `plugins`, `python_version`) but **no `to_dict()` / JSON
serializer**, so rendering it generically (e.g. dumping to JSON) takes manual
field access.

**Why it matters.** Minor, but the discovery flow already hands you a
`ModelRegistry`; a doctor that re-discovers by `group` name feels disconnected,
and the missing serializer means every consumer hand-rolls report rendering.

**Suggested improvement.** Accept an optional `registry` (or `eps`) so doctor can
reuse an already-discovered set, and add `DoctorReport.model_dump()` /
`to_dict()` (it's pydantic-shaped already) for generic rendering.

---

## 9. [doc] Backpressure coalescing silently drops interim `partial`s when the consumer is fast — surprising for a "show partials" app

**What happened.** Feeding a streaming session quickly (or a scripted engine that
emits its whole script in a tight loop), my live UI saw **zero** `partial`
events: the protocol's `_CoalescingBuffer` correctly merges a pending partial into
its same-segment `final` under backpressure (spec ST.6.4). This is *correct and
desirable* behavior — but for a captioning app whose entire point is to *show*
interim partials, it was initially baffling ("why no partials?"). Partials only
become observable when events arrive with enough spacing that the consumer drains
between them.

**Why it matters.** An app developer reading "engines emit `partial` events"
reasonably expects to receive them; discovering that fast consumption legitimately
elides them (and that this is a feature) took debugging. It's a conceptual gap
between "the engine emitted a partial" and "my loop observed one."

**Suggested improvement.** Document in the streaming guide that `partial`
coalescing means **interim partials are a best-effort display aid, not a
guaranteed event count** — an app must treat the latest `partial`/`final` as
truth and never count on seeing every partial. (My scripted test engine had to
add an inter-event delay to make partials observable; that lesson belongs in the
docs.)

---

## 10. [papercut] No public helper to reconcile the live view against `session.result()`

**What happened.** A live UI must keep its own view state (it needs in-progress
`partial` text + the `stable_until` boundary, which `session.result()` — committed
finals only — doesn't expose). At `done` I wanted to assert "my committed view ==
the authoritative result." They differ by whitespace (finding §4) and by my
per-segment `strip()`, so an exact equality check is brittle. There's no
canonical "normalized transcript text" helper to compare against.

**Why it matters.** Every serious streaming app maintains a parallel view model
and will want to sanity-check it against the reducer. Without a normalization
helper, each app invents its own.

**Suggested improvement.** A `TranscriptionResult.normalized_text` (whitespace-
collapsed) or a documented note that `result().text` is the single source of
truth for the final transcript and apps should display *that* at completion
(which is what I do for export), keeping their live view purely for the in-flight
display.

---

## A. What worked really well (credit where due)

These made the app *much* easier than integrating raw ASR engines, and are worth
preserving:

- **Zero-config discovery is genuinely zero-config.** `discover_models()` →
  `names()` → `engine_class(key).properties / .declared_capabilities` let me list
  engines, show their full capability tree, and render a settings form **without
  instantiating** anything — exactly what a credentialed engine needs. This is the
  headline promise and it delivers.
- **The streaming event model is complete and the reducer is tiny.** The canonical
  `partial`/`final`/`supersede` reduce is ~15 lines; my whole view reducer is
  ~120. `TranscriptionEvent` exposes `stable_text`, `is_content`, `is_terminal`
  as public helpers — I used all three. The factory classmethods
  (`TranscriptionEvent.partial/final/closed/supersede/progress/done/make_error`)
  made the scripted test engine pleasant to write.
- **The `_LifecycleGuard` is a fantastic safety net for app + engine authors
  alike.** My first scripted script had inconsistent `stable_until` math; the guard
  *suppressed* the offending events and emitted precise
  `frozen_prefix_rewritten_supersede` diagnostics naming the diverging
  prefixes. That turned a subtle spec violation into an obvious, debuggable
  message — and proved the app never has to defend against a non-compliant engine.
- **`session.diagnostics()` is available from frame 1** (before iterating), so a
  diagnostics panel can render immediately. The diagnostics carry `param` /
  `provided` / `effective`, which made them trivially presentable.
- **`AudioFormat` rejection messages are exemplary.** Wrong encoding, non-mono,
  and unreachable sample rate each raise `UnsupportedFeatureError` with a message
  that names the offending value, the engine's allowed set, the reason, *and* the
  fix. I surfaced them verbatim.
- **`to_srt` / `to_vtt` for free.** Both cookbook engines and my scripted engine
  got correct, security-hardened subtitle export with one call each. Real win.
- **Secret-field markers in the JSON Schema** (`secret` / `writeOnly` /
  `format: password`, including inside `anyOf` for `SecretStr | None`) let me
  detect credentials generically and never echo them. Clean.

---

## B. Smallest changes with the biggest DX payoff

1. **§2** — export `BaseProperties.preferred_wire_format()` (already exists,
   privately). Removes the #1 streaming-app stumbling block.
2. **§3** — add `engine.can_stream_incrementally()` and/or make `supports()`
   reject unknown dot-paths. Kills a silent-wrong-answer footgun.
3. **§1** — export `CANONICAL_WIRE_ENCODING = "pcm_s16le"`.
4. **§5** — wrap config `ValidationError` in `ConfigError` (one error type to
   catch) or document the dual-catch.
5. **§4** — trim per-segment whitespace in `StreamReducer.result()`.
