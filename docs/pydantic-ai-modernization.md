# Pydantic AI Modernization — Curated Diff (1.72.0 → 1.96.x)

> Curated 2026-05-15. Sources: GitHub releases, `pydantic.dev/docs/ai/*` (docs moved from
> `ai.pydantic.dev`), the `pydantic-ai-harness` repo/PyPI, and a full audit of this repo's demos.
> This is a *curation pass* — a map of what's substantial and worth adopting, before any code changes.

---

## 0. TL;DR

1. **We pin `pydantic-ai==1.72.0`** (released 2026-03-26). **Latest is `1.96.1`** (released today,
   2026-05-15). We are ~24 minor releases behind.
2. **"Agentic Harness" is not a literal product name.** It's an umbrella for two things:
   - **Capabilities + Hooks API** — in *core*, stable, **introduced in v1.71.0**. We already have
     it on 1.72.0; we just don't use it anywhere.
   - **`pydantic-ai-harness` package** — the optional "batteries" (CodeMode, memory, guardrails,
     sub-agents). Beta (0.3.0). **Requires `pydantic-ai-slim>=1.95.1`** — we cannot install it on
     1.72.0.
3. **Good news: we have essentially no deprecated code to let go of.** The audit found zero usage
   of `result.data`, `result_type=`, `instrument=`, `history_processors=`, `prepare_tools=`,
   `builtin_tools=`, `system_prompt=`, method-style `result.usage()`, or bare `openai:` strings.
   Our demos already use the modern surface (`instructions=`, `output_type=`, `result.output`,
   `run_stream_events()`, explicit `OpenAIChatModel`/`OpenAIProvider`).
4. **The real gap is coverage, not rot.** As a *learning* repo, we have no demos for the headline
   modern concepts: Capabilities, Hooks, tools/toolsets, dependencies/`RunContext`, the harness
   package/CodeMode, testing, or evals.
5. **Decision needed: bump the version.** Staying on 1.72.0 blocks the headline feature (the
   harness package) and ~24 releases of enhancements. Bumping is low-risk *because* we use no
   deprecated APIs.
6. **The `/pydantic-ai` skill itself is partly stale** (old `gpt-4o` model strings, no
   streaming/`UIEventStream` doc, graphs.md inconsistencies) — flagged per CLAUDE.md.

---

## 1. Version situation

| | Version | Released | Notes |
|---|---|---|---|
| We pin | `pydantic-ai >=1.72.0`, locked `1.72.0` | 2026-03-26 | One release *after* the Capabilities rework (v1.71.0) |
| Latest core | `1.96.1` | 2026-05-15 | The 1.9x line is explicit "v2 prep" — many deprecations |
| Harness pkg | `pydantic-ai-harness 0.3.0` | 2026-05-13 | Beta. Needs `pydantic-ai-slim>=1.95.1` |
| v2 | — | ~June 2026 (signposted) | Deprecated APIs removed then |

**What we already have on 1.72.0:** the Capabilities/Hooks API (landed v1.71.0), Agent Specs
(`Agent.from_file`), provider-adaptive tools, `args_validator` on tools, `description=` on `Agent`.

**What we're missing by staying on 1.72.0:** the `pydantic-ai-harness` package (CodeMode etc.),
native tool search (v1.95), `tool_choice` setting (v1.93), `conversation_id` correlation (v1.89),
server-side compaction (v1.80/1.84), optional/nullable `output_type` (v1.83), Gemini-3
tools+structured-output combo (v1.95), plus the v1.95–1.96 deprecation warnings that would *teach*
us the modern surface.

---

## 2. The "Agentic Harness" demystified

Pydantic AI's framing: *"capabilities and hooks are how you give an agent its **harness** —
bundles of tools, lifecycle hooks, instructions, and model settings."* It is **additive
configuration**, not a replacement for `Agent` / `run()` / `run_sync()` / `run_stream()`.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking, WebSearch

agent = Agent(
    "anthropic:claude-opus-4-7",
    instructions="You are a research assistant.",
    capabilities=[
        Thinking(effort="high"),
        WebSearch(local="duckduckgo"),   # provider-native, explicit local fallback
    ],
)
```

Two layers:

- **Core `pydantic_ai.capabilities`** (stable, since v1.71.0 — we have it): `Thinking`,
  `WebSearch`, `WebFetch`, `MCP`, `ImageGeneration`, `Hooks`, `Instrumentation`, `ToolSearch`,
  `NativeTool`, `PrepareTools`/`PrepareOutputTools`, `ProcessHistory`, `ProcessEventStream`,
  `HandleDeferredToolCalls`. Composable (before-hooks in order, after-hooks reversed; ordering
  via `CapabilityOrdering`). Custom capabilities subclass `AbstractCapability`. Capabilities can
  be dynamic (a callable resolved per run, since v1.89).
- **`pydantic-ai-harness` package** (beta 0.3.0, needs `>=1.95.1`): the flagship is **`CodeMode`**,
  which wraps *all* tools into one sandboxed `run_code` tool (Monty sandbox) so the model can
  orchestrate N tool calls in a single round-trip (e.g. "fetch 10 items, process each" → 1 model
  call instead of 11). Memory, guardrails, context compaction, sub-agents/teams are in-progress in
  this package — **not in core**.

**`Hooks`** = decorator sugar over ~20 lifecycle interception points, contract "raise to
propagate, return to recover":

```python
from pydantic_ai.capabilities import Hooks

hooks = Hooks()

@hooks.on.before_model_request
async def log_request(ctx, request_context):
    print(f"Sending {len(request_context.messages)} messages")
    return request_context

agent = Agent("openai-chat:gpt-5.2", name="my_agent", capabilities=[hooks])
```

**When NOT to reach for it:** simple prompt → structured output with a couple of tools (plain
`Agent` is enough — matches our "keep POCs minimal" guideline). And the harness *package* is beta;
core capabilities are stable.

---

## 3. Substantial new features worth learning / demoing

Prioritized for a learning repo. (P1 = headline, demo first.)

| Pri | Feature | Version | Why it matters |
|---|---|---|---|
| **P1** | **Capabilities API** (`capabilities=[...]`) | v1.71.0 | The single modern configuration surface; replaces scattered constructor kwargs. We use none. |
| **P1** | **Hooks** (lifecycle interception) | v1.71.0+ | Idiomatic replacement for custom agent wrappers/middleware. |
| **P1** | **Function tools** `@agent.tool` / `tool_plain`, **toolsets**, **deferred/approval (HITL)** tools | core | We have a `demos/tools/` category with **no tool demos at all**. Foundational. |
| **P1** | **Dependencies / `RunContext` / `deps_type`** | core | Foundational DI pattern, completely absent from our demos. |
| **P2** | **`pydantic-ai-harness` + `CodeMode`** | pkg 0.3.0 (needs core ≥1.95.1) | The headline "harness" feature; big round-trip savings. **Needs version bump.** |
| **P2** | **Native Tool Search** (`ToolSearch`, `defer_loading`) | v1.95.0 | Scales large toolsets without prompt bloat. |
| **P2** | **Thinking** capability (cross-provider reasoning) | v1.71.0 | Clean cross-provider reasoning effort control. |
| **P2** | **Testing**: `TestModel`, `FunctionModel`, `agent.override`, `capture_run_messages` | core | We have zero tests/test demos; cheap, deterministic. |
| **P3** | **Evals** (`pydantic_evals`: `Dataset`/`Case`, LLM-judge) | core | Already a locked dep (`pydantic-evals 1.72.0`); unused. |
| **P3** | **Message history & serialization** (`message_history=`, `ModelMessagesTypeAdapter`) | core | Conversational state — not demoed. |
| **P3** | **`tool_choice` setting** + output-tool stream events | v1.93.0 | Finer tool-invocation control + full stream observability. |
| **P3** | **Optional/nullable `output_type`** (`str | None`), `template=False` | v1.64/1.83 | Cleaner structured-output modeling. |
| **P3** | **Multi-agent delegation w/ usage rollup** (`usage=ctx.usage`), hand-off, A2A | core | We do routing + graph orchestration but not agent-as-tool delegation/A2A. |
| **P3** | **`conversation_id`** + server-side compaction | v1.89/1.80 | Native conversation state vs manual history juggling. |

---

## 4. Deprecated / old patterns — status in *our* code

**Headline: the audit found none of these in our codebase.** This section is therefore mostly a
*forward-looking avoid-list* (don't introduce these in new demos) plus grep guards.

| Deprecated pattern | Replace with | Since | In our code? |
|---|---|---|---|
| `result.data` | `result.output` | v0.1/v0.6 | ❌ none (we use `result.output`) |
| `result_type=`, `result_retries=` | `output_type=`, `output_retries=` | v0.6 | ❌ none |
| `result.usage()`, `result.timestamp()`, `stream.get()` (methods) | `.usage`, `.timestamp`, `stream.response` (properties) | v1.96.0 | ❌ none |
| `Agent(instrument=...)` | `capabilities=[Instrumentation()]` *(setter un-deprecated in 1.95.1; both work)* | v1.95/96 | ❌ none |
| `Agent(history_processors=...)` | `capabilities=[ProcessHistory(...)]` | v1.96.0 | ❌ none |
| `Agent(prepare_tools=...)` / `prepare_output_tools=` / `event_stream_handler=` | matching capabilities | v1.96.1 | ❌ none |
| `prepare_tools` touching output tools (semantics changed) | `prepare_output_tools` | v1.88.0 (breaking) | ❌ none |
| `builtin_tools=` / "built-in tools" term | `capabilities=[NativeTool(...)]` / `WebSearch(...)` ("native tools") | v1.95.0 | ❌ none |
| Implicit provider auto-fallback | explicit `local=` / `native=` | v1.95.0 | ❌ none |
| Bare `openai:` model prefix | `openai-chat:` or `openai-responses:` | v1.96.0 | ❌ none (we pass an explicit `OpenAIChatModel`) |
| `AGUIApp`, `Agent.to_ag_ui()`, `pydantic_ai.ag_ui` | `AGUIAdapter` | v1.96.0 | ❌ none |
| `OutlinesModel` / `OutlinesProvider` | native/prompted output | removed ~v1.96 | ❌ none |
| `@agent.system_prompt` / `system_prompt=` | `instructions=` / `@agent.instructions` (preferred, not forced) | — | ❌ none (we already use `instructions=`) |

**Grep guard** (run before commits to keep new demos clean):

```
result\.data\b | \bresult_type= | \bresult_retries= | result\.usage\(\) | result\.timestamp\(\)
stream\.get\(\) | \binstrument= | history_processors= | prepare_tools= | prepare_output_tools=
event_stream_handler= | builtin_tools= | to_ag_ui\(|AGUIApp|pydantic_ai\.ag_ui
OutlinesModel|OutlinesProvider | Agent\(['"]openai: | @agent\.system_prompt|system_prompt=
```

One nuance to verify when we bump: **`prepare_tools` was a breaking change in v1.88.0** (now
function-tools only; output-tool logic must move to `prepare_output_tools`). We don't use it, so
no action — just noted.

---

## 5. `/pydantic-ai` skill staleness (per CLAUDE.md)

The codebase audit cross-checked `~/.claude/skills/pydantic-ai/`. Issues worth a skill-update
sub-agent:

1. **No streaming/UI doc.** Our most advanced demo (`demos/streaming/graph_streaming.py`) uses
   `pydantic_ai.ui.UIEventStream`, `run_stream_events()`, `streaming_response()`,
   `transform_stream()`, `AgentRunResultEvent` — none documented in the skill.
2. **Stale model strings.** `tools.md`/`output.md`/`messages.md`/`dependencies.md`/`testing.md`
   use `'openai:gpt-4o'`; SKILL.md mixes `claude-sonnet-4-6` with `claude-sonnet-4-0`. Normalize
   to a current family and `openai-chat:`/`openai-responses:`.
3. **`graphs.md` disagrees with working code:** uses `Graph(nodes=[...], state_type=State)` while
   our verified-working demos use `Graph(nodes=[...])`; also `print(result)` vs `result.output`.
4. **Import-path inconsistencies:** `ModelSettings` imported from both `pydantic_ai.models` and
   `pydantic_ai.settings`; `ModelMessagesTypeAdapter` path unverified.
5. Capabilities / Agent Specs / A2A / `from_file` sections are unverified against 1.72.0.

---

## 6. Recommended next steps (not yet executed)

1. **Bump `pydantic-ai` to latest `1.96.x`** (and `pydantic-evals`). Low-risk: zero deprecated
   usage. Unlocks the harness package + ~24 releases of features. `uv sync`, run all demos,
   eyeball any new deprecation warnings.
2. **Fill the demo gaps**, suggested order:
   `demos/tools/` (function tools, toolsets, approval/HITL) → `demos/dependencies/` (`RunContext`)
   → `demos/capabilities/` (Thinking, Hooks, custom `AbstractCapability`) →
   `demos/harness/` (`CodeMode`) → `demos/testing/` (`TestModel`) → `demos/evals/`.
3. **Spawn a skill-update sub-agent** to fix the `/pydantic-ai` skill staleness in §5.
4. Add the §4 grep guard (optional pre-commit) so new demos stay on the modern surface.

---

## 7. Execution log — DONE (2026-05-15)

All of §6 was executed.

**Version bump:** `pyproject.toml` → `pydantic-ai>=1.96.1`,
`pydantic-ai-harness[code-mode]>=0.3.0`, `pydantic-evals>=1.96.1`. `uv sync` →
pydantic-ai/slim/evals/graph **1.96.1**, harness **0.3.0**, pydantic-monty **0.0.17**. All 6
pre-existing demos still compile and run — zero regressions (the "no deprecated APIs" finding
held). `src/core/models.py` default model updated to `qwen/qwen3.6-35b-a3b` (the old default
wasn't loaded in LM Studio; new one verified for tools + structured output + thinking).

**New demos built & verified** (each runs with clear labeled stdout):

| Dir | Files | Verified |
|---|---|---|
| `demos/tools/` | `function_tools.py`, `toolsets.py`, `human_in_the_loop.py` | live LLM |
| `demos/dependencies/` | `dependency_injection.py`, `override_deps.py` | live LLM |
| `demos/capabilities/` | `thinking.py`, `hooks.py`, `custom_capability.py` | live LLM |
| `demos/harness/` | `code_mode.py` | live LLM (9 tool calls in 1 round-trip) |
| `demos/testing/` | `test_model.py`, `function_model.py` | offline, deterministic |
| `demos/evals/` | `basic_eval.py` | offline, deterministic |

**Skill repaired:** `~/.claude/skills/pydantic-ai/` modernized across 17 files + new
`streaming.md` (3 sub-agents, disjoint file sets, verified against installed 1.96.1 source).

### API corrections discovered while building (real 1.96.1 surface)

These are things even the official docs/old skill got wrong — captured here as the authoritative
reference for future demos:

- `Tool(...)` uses **`max_retries=`**, not `retries=` (old form raises `TypeError`).
- **No `@agent.output_function` decorator** — pass output functions via `output_type=`.
- **No `@agent.run_metadata` decorator / no `run_metadata=` kwarg** — it's `metadata=`;
  `RunContext` exposes `ctx.metadata` (not `ctx.run_metadata`).
- **No `Toolset` export** — use `FunctionToolset` (+ `.filtered()`/`.prefixed()`/…) via
  `toolsets=[...]`. `.prefixed("bank")` joins with `_` → `bank_get_rate`.
- HITL: modern flow is `output_type=[str, DeferredToolRequests]` + `requires_approval=True`,
  resume via `deferred_tool_results=`; helper `DeferredToolRequests.build_results(approvals=...)`.
  The old `node.tool_calls_requiring_approval` / `call.approve()` API does not exist.
- `BinaryImage` is imported from `pydantic_ai` (top-level), not `pydantic_ai.output`.
- `ModelSettings` canonically lives in `pydantic_ai.settings` (re-exported top-level).
- Capabilities: `pydantic_ai.capabilities` exports ~23 classes; `BuiltinTool`→`NativeTool`,
  `HistoryProcessor`→`ProcessHistory` are deprecated aliases. `Thinking(effort=...)` where
  effort ∈ `bool | 'minimal'|'low'|'medium'|'high'|'xhigh'`; sole effect is
  `model_settings={'thinking': effort}` (silently ignored on always-on reasoning models).
- Hooks: decorator entrypoint is `hooks.on.<name>`; names drop `wrap_`/`on_` prefixes
  (`wrap_run`→`run`, `on_run_error`→`run_error`); tool hooks filter via `tools=` (not
  `tool_name=`); `SkipModelRequest` is in `pydantic_ai.exceptions`. Contract: "raise to
  propagate, return to recover". `after_output_validate` does **not** fire for plain text —
  use `after_model_request` for a plain-text output guardrail.
- `pydantic_graph.Graph(*, nodes, name=None, state_type=<inferred>, run_end_type=<inferred>,
  auto_instrument=True)` — keyword-only, **no `deps_type=`** (deps come from `BaseNode`'s 2nd
  generic). Classic `graph.run()/run_sync()` returns `GraphRunResult` (`.output`/`.state`/
  `.persistence`) → use `result.output`; the **beta** graph API's `run()` returns the output
  directly.
- Testing: `ALLOW_MODEL_REQUESTS` is a **module attribute**
  (`pydantic_ai.models.ALLOW_MODEL_REQUESTS = False`), not an env var. `AgentInfo` exposes
  `function_tools`/`output_tools` (not `tools`/`output_type`/`deps_type`).
- Evals: `Dataset`/`Case` are keyword-only; `Dataset` wants explicit `name=`. The `Python`
  evaluator was **removed for security**. Judge helpers live in
  `pydantic_evals.evaluators.llm_as_a_judge`.
- CodeMode: `from pydantic_ai_harness import CodeMode`; `CodeMode(tools='all', max_retries=3)`;
  attach via `capabilities=[CodeMode()]`; collapses all tools into one sandboxed `run_code`
  (Monty); needs the `code-mode` extra + `pydantic-ai-slim>=1.95.1`.

---

## 8. Bespoke vs Official Skill Comparison & Quality Charter (2026-05-19)

We compared our hand-curated `~/.claude/skills/pydantic-ai/` against the official Pydantic-team
skill (`building-pydantic-ai-agents` **v1.1.0**). This section records the verdict, the one hard
contradiction it resolved, our moat, and the quality charter that governs the bespoke skill going
forward. (The bespoke skill itself lives outside this repo and is owned by other agents — this is
the *curation record*, not the skill content.)

### 8.1 Verified fact (re-checked against installed source)

`pydantic_ai` installed = **1.96.1**. `AgentRunResult.usage` is a
`_DeprecatedCallableProperty` (from `pydantic_ai._deprecated_callable`). Therefore
**`result.usage` (property) is the correct/current form**, and **`result.usage()` (call) is the
deprecated shim**. This was verified by direct introspection of the installed package, not docs.

### 8.2 Comparison summary

| | Bespoke `~/.claude/skills/pydantic-ai/` | Official `building-pydantic-ai-agents` v1.1.0 |
|---|---|---|
| Size | ~4,650 lines / 18 files — deep per topic | ~1,450 lines / 11 files — broad, shallow |
| Version discipline | Every file stamped "Verified against pydantic-ai 1.96.1"; explicit deprecation→replacement maps (`instrument=`→`Instrumentation`, `builtin_tools=`→`NativeTool`, `MCPServerHTTP`→`MCPServerStreamableHTTP`, AG-UI removal map) | Unpinned; uses deprecated `.usage()` and bare `openai:` strings as if current |
| Hard contradiction | `result.usage` **property** (correct) | `result.usage()` **method** (deprecated) |

The single hard contradiction (`result.usage` property vs `.usage()` method) is resolved
**decisively in the bespoke skill's favor** against installed 1.96.1 (see §8.1).

### 8.3 Bespoke moat — do not regress

1. **Version discipline** — per-file "Verified against pydantic-ai X.Y.Z" stamps + explicit
   deprecation→replacement maps.
2. **Deep harness/capabilities coverage** — `AbstractCapability` authoring, `CapabilityOrdering`,
   `for_run` isolation, `pydantic-ai-harness`/`CodeMode` (beta).
3. **Unique advanced content the official skill lacks** — beta `GraphBuilder` with reducers, the
   hooks naming-convention decoder table, nested-graph streaming fan-in (anyio), the `Embedder`
   testing surface, agent specs with Handlebars / `RunContext.deps` binding.
4. **Pervasive anti-hallucination callouts** — explicit "this does NOT exist" notes.

### 8.4 Official skill — strengths and weaknesses

- **Genuine advantage (port as structure only, never as content):** navigation — decision trees,
  front-loaded gotchas, intent-keyed routing. This is being ported into the bespoke skill as
  *structure only*.
- **Weaknesses:** future/fictional model names with no caveat; one internally-inconsistent graph
  example despite a "tested examples" claim; asserted-but-undemonstrated APIs (`run_stream_sync`,
  exact hook signatures); a 108-line redirect-only `COMMON-TASKS.md` carried as dead weight.

### 8.5 Quality charter (governs the bespoke skill)

1. **The bespoke skill is the source of truth** for Pydantic AI in our work. The official skill
   is a **structural reference only — never a content source** (it lags and presents deprecated
   patterns as current).
2. **Verify before encoding.** Every claim/example is checked against the *actually installed*
   `pydantic_ai` version; each reference file carries a "Verified against pydantic-ai X.Y.Z"
   stamp.
3. **Maintain explicit deprecation→replacement maps.** Prefer the modern surface: capabilities
   over constructor kwargs; properties over deprecated callables; `openai-chat:` /
   `openai-responses:` over bare `openai:`.
4. **Depth over breadth.** Cover the hard / bleeding-edge surface (harness/capabilities, graphs,
   streaming) thoroughly rather than skimming everything.
5. **Use anti-hallucination "this does NOT exist" callouts deliberately.**
6. **On every framework bump, re-verify** stamped files against the new version and update the
   deprecation maps.
