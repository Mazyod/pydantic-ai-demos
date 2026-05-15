"""Toolsets: bundle tools, then compose with transforms before attaching.

Concept:
  A `FunctionToolset` is a reusable, named bundle of tools, independent of any
  agent. It supports composition transforms that return NEW toolsets without
  mutating the original:
    - `.filtered(fn)`  -> hide tools where fn(ctx, tool_def) is False
    - `.renamed({...})`-> expose tools under different names
    - `.prefixed("x")` -> namespace every tool name with a prefix
    - `.combined(...)` (module-level CombinedToolset) -> merge bundles
  Attach the (transformed) toolset(s) via `Agent(..., toolsets=[...])`.

Goal / what to observe:
  We build a 3-tool bundle, then `.filtered()` out the dangerous one and
  `.prefixed("bank_")` the rest. The printed tool list proves the agent only
  sees the transformed surface; the answer uses a prefixed tool.
"""

from pydantic_ai import Agent, FunctionToolset, RunContext

from core import get_model

# A standalone, agent-independent bundle of tools.
banking = FunctionToolset(id="banking")


@banking.tool_plain
def get_rate(currency: str) -> float:
    """Return a fixed demo FX rate from USD to the given currency code."""
    rates = {"EUR": 0.92, "GBP": 0.79, "JPY": 156.0}
    rate = rates.get(currency.upper(), 1.0)
    print(f"  [tool call] get_rate(currency={currency!r}) -> {rate}")
    return rate


@banking.tool_plain
def list_branches() -> list[str]:
    """List the bank's branch cities."""
    print("  [tool call] list_branches()")
    return ["Kuwait City", "Dubai", "London"]


@banking.tool_plain
def wire_transfer(to_account: str, amount_usd: float) -> str:
    """Irreversibly wire money. (Dangerous - filtered out below.)"""
    print(f"  [tool call] wire_transfer({to_account=}, {amount_usd=})")
    return f"Wired ${amount_usd} to {to_account}."


def _safe_only(_ctx: RunContext[None], tool_def) -> bool:
    """Filter predicate: drop the dangerous wire_transfer tool."""
    return tool_def.name != "wire_transfer"


# Compose: hide the dangerous tool, then namespace the survivors.
safe_banking = banking.filtered(_safe_only).prefixed("bank")

agent = Agent(
    get_model(),
    toolsets=[safe_banking],
    instructions="You are a banking assistant. Use tools to answer precisely.",
)


if __name__ == "__main__":
    print("=== Toolsets composition demo ===")
    print("Original bundle 'banking' tools : get_rate, list_branches, wire_transfer")
    print("Transform applied               : .filtered(safe_only).prefixed('bank')")
    print("Expected agent-visible surface  : bank_get_rate, bank_list_branches")
    print("(wire_transfer filtered out; survivors namespaced with 'bank_')\n")

    print("Question: What is the USD to GBP rate?\n")
    print("Tool activity:")
    result = agent.run_sync("What is the USD to GBP rate? Use the tool.")

    print("\n--- Final answer ---")
    print(result.output)

    print("\n--- Tool calls made (note the 'bank_' prefix) ---")
    for msg in result.all_messages():
        for part in msg.parts:
            if part.part_kind == "tool-call":
                print(f"  - {part.tool_name}({part.args})")
