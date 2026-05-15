"""Harness "Code Mode" — orchestrate many tool calls in ONE model round-trip.

CONCEPT
    `pydantic-ai-harness` ships the `CodeMode` capability. Added to an agent via
    `capabilities=[CodeMode()]`, it wraps *every* registered tool behind a single
    sandboxed meta-tool called `run_code`. Instead of the model emitting one tool
    call per step (call get_weather, wait, call get_weather again, wait, call
    convert_temp, wait...), the model writes a *Python program* once. That program
    calls the wrapped tools as plain functions — in loops, with comprehensions,
    doing arithmetic — inside a Monty sandbox (a safe Python subset). The whole
    orchestration happens in a single model -> tool -> model round-trip.

GOAL
    Register 3 trivial `@agent.tool_plain` tools and ask a question that needs
    *several* of them combined ("weather for 4 cities, in Celsius, plus the
    average"). Show that the model issued exactly ONE `run_code` tool call, and
    that the real per-tool calls happened *inside* the sandbox (recorded in the
    `run_code` return part's metadata).

WHAT TO OBSERVE
    1. The final answer combines data from many tool invocations.
    2. The model's message history contains ONE `run_code` ToolCallPart — not
       N separate get_weather / convert_temp calls.
    3. The nested calls the sandbox actually made are visible in the run_code
       ToolReturnPart metadata (`code_mode`, `tool_calls`, `tool_returns`),
       proving the loop ran sandbox-side, not as model round-trips.

REAL API (verified against pydantic-ai 1.96.1 / pydantic-ai-harness 0.3.0 source)
    - `from pydantic_ai_harness import CodeMode`  (lazy export; needs the
      `code-mode` extra, i.e. `pydantic-monty`).
    - `CodeMode(tools='all', max_retries=3)` — a dataclass capability.
      `tools='all'` sandboxes every tool; pass a name list / predicate to keep
      some tools as normal native calls.
    - Attach via the modern surface only: `Agent(model, capabilities=[CodeMode()])`.
    - The wrapped tools collapse into one tool literally named `run_code`. Its
      ToolReturnPart carries `metadata={'code_mode': True, 'tool_calls': {...},
      'tool_returns': {...}}` — that is how we make the lesson visible here.
    - Give sandboxed tools explicit return type hints: CodeMode warns when a tool
      has no return schema (signature shows `-> Any`, hurting effectiveness).

OBSERVED BEHAVIOR WITH THE LOCAL MODEL (qwen3.6-35b on LM Studio)
    The local model reliably chooses `run_code` and writes a single Python
    snippet that loops over the cities and calls both tools, returning the
    combined result in ONE round-trip — exactly the CodeMode payoff. Run output
    is labeled below. Caveat: smaller local models occasionally need CodeMode's
    built-in retry (syntax/type errors are fed back automatically, up to
    `max_retries`); the demo keeps the task small to minimize LLM calls and
    make a clean run likely. If a given run's wording drifts, the structural
    lesson (one `run_code` call wrapping N sandbox-side tool calls) still holds
    and is asserted/printed explicitly.
"""

from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai_harness import CodeMode

from core import get_model

agent = Agent(
    get_model(),
    instructions=(
        "You answer questions using the available tools. "
        "Be concise and report concrete numbers."
    ),
    capabilities=[CodeMode()],
)

# Tiny fake "data sources". Note the explicit return type hints: CodeMode uses
# them to give the model a typed function signature inside the sandbox.
_WEATHER_F = {
    "kuwait city": 104.0,
    "london": 59.0,
    "tokyo": 71.0,
    "reykjavik": 41.0,
}


@agent.tool_plain
def get_weather(city: str) -> float:
    """Return the current temperature for a city, in degrees Fahrenheit."""
    return _WEATHER_F[city.strip().lower()]


@agent.tool_plain
def fahrenheit_to_celsius(f: float) -> float:
    """Convert a temperature from Fahrenheit to Celsius."""
    return round((f - 32) * 5 / 9, 1)


@agent.tool_plain
def list_cities() -> list[str]:
    """Return the list of cities weather data is available for."""
    return list(_WEATHER_F)


def main() -> None:
    question = (
        "For every city you have weather data for, get its temperature, "
        "convert it to Celsius, and report each city's Celsius temperature "
        "plus the average Celsius across all cities (one decimal place)."
    )

    print("=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)

    result = agent.run_sync(question)

    print()
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result.output)

    # ---- Make the lesson visible: inspect the message history -------------
    messages = result.all_messages()

    run_code_calls: list[ToolCallPart] = []
    other_tool_calls: list[ToolCallPart] = []
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, ToolCallPart):
                if part.tool_name == "run_code":
                    run_code_calls.append(part)
                else:
                    other_tool_calls.append(part)

    print()
    print("=" * 70)
    print("THE LESSON: ONE round-trip, not many")
    print("=" * 70)
    print(f"run_code tool calls (model round-trips):     {len(run_code_calls)}")
    print(f"other native tool calls (model round-trips):  {len(other_tool_calls)}")
    if other_tool_calls:
        names = ", ".join(sorted({c.tool_name for c in other_tool_calls}))
        print(
            f"  (the model also probed [{names}] directly before writing\n"
            f"   the loop — still tiny vs. one round-trip per data point)"
        )
    print(
        "  -> With CodeMode, the tools collapse into the single `run_code`\n"
        "     meta-tool; the bulk orchestration ran as model-written Python."
    )

    # The Python the model wrote and ran in the Monty sandbox.
    for i, call in enumerate(run_code_calls, 1):
        args = call.args_as_dict()
        code = args.get("code", "")
        print()
        print("-" * 70)
        print(f"SANDBOXED PYTHON the model wrote (run_code call #{i})")
        print("-" * 70)
        print(code.strip())

    # The real per-tool calls happened INSIDE the sandbox. CodeMode records
    # them on the run_code ToolReturnPart metadata.
    for msg in messages:
        for part in msg.parts:
            if (
                isinstance(part, ToolReturnPart)
                and part.tool_name == "run_code"
                and isinstance(part.metadata, dict)
                and part.metadata.get("code_mode")
            ):
                nested = part.metadata.get("tool_calls", {})
                print()
                print("-" * 70)
                print("ACTUAL tool calls made INSIDE the sandbox")
                print("-" * 70)
                print(f"count: {len(nested)} (these were NOT model round-trips)")
                for tc in nested.values():
                    print(f"  - {tc.tool_name}({tc.args_as_dict()})")

    # A crisp, explicit statement of the structural lesson.
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    nested_total = sum(
        len(part.metadata.get("tool_calls", {}))
        for msg in messages
        for part in msg.parts
        if isinstance(part, ToolReturnPart)
        and part.tool_name == "run_code"
        and isinstance(part.metadata, dict)
        and part.metadata.get("code_mode")
    )
    print(
        f"{nested_total} tool invocations were orchestrated by the model in just "
        f"{len(run_code_calls)} `run_code` round-trip(s).\n"
        "Without CodeMode that would have been roughly one model round-trip "
        "per tool call."
    )


if __name__ == "__main__":
    main()
