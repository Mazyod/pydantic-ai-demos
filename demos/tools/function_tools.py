"""Function tools: `@agent.tool` (with RunContext) vs `@agent.tool_plain` (no ctx).

Concept:
  The model can call Python functions you register as tools. Two decorators:
    - `@agent.tool`       -> first arg is `RunContext[Deps]`; use it to read
                             per-run dependencies (here: a user's account).
    - `@agent.tool_plain` -> no context; a pure utility the model can call.

Goal / what to observe:
  Ask one question that forces BOTH tools to fire. The printed tool-call log
  proves the model actually invoked them, and the final answer is derived
  from the tool return values (not hallucinated).
"""

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from core import get_model


@dataclass
class Account:
    """Per-run dependency injected via `deps=`."""

    owner: str
    balance_usd: float


agent = Agent(
    get_model(),
    deps_type=Account,
    instructions=(
        "You are a banking assistant. Use the available tools to answer. "
        "Never guess balances or rates - always call the tools."
    ),
)


@agent.tool
def get_balance(ctx: RunContext[Account]) -> str:
    """Return the signed-in customer's current balance in USD."""
    print(f"  [tool call] get_balance() -> owner={ctx.deps.owner}")
    return f"{ctx.deps.owner} has a balance of ${ctx.deps.balance_usd:,.2f} USD."


@agent.tool_plain
def usd_to_eur(amount_usd: float) -> float:
    """Convert a USD amount to EUR at a fixed demo rate of 0.92."""
    result = round(amount_usd * 0.92, 2)
    print(f"  [tool call] usd_to_eur(amount_usd={amount_usd}) -> {result}")
    return result


if __name__ == "__main__":
    deps = Account(owner="Alice", balance_usd=1500.00)

    print("=== Function tools demo ===")
    print("Deps injected: Account(owner='Alice', balance_usd=1500.00)")
    print("Question: What is my balance, and what is it worth in euros?\n")

    print("Tool activity:")
    result = agent.run_sync(
        "What is my balance, and what is it worth in euros?",
        deps=deps,
    )

    print("\n--- Final answer (derived from tool returns) ---")
    print(result.output)

    print("\n--- Tools the model actually invoked ---")
    for msg in result.all_messages():
        for part in msg.parts:
            if part.part_kind == "tool-call":
                print(f"  - {part.tool_name}({part.args})")
