"""Dependency injection with RunContext.

Demonstrates: a dataclass deps object passed via `deps_type=`, read inside a
tool through `RunContext[Deps]` (`ctx.deps`), AND a dynamic
`@agent.instructions` function that uses `ctx.deps` to personalize the system
instruction.

Goal: show the SAME agent producing different behavior purely because the
injected dependencies differ -- no agent/code changes between runs.
"""

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from core import get_model

model = get_model()


# Dependencies are a plain dataclass. Nothing framework-specific here.
@dataclass
class CustomerCtx:
    name: str
    tier: str  # "free" or "pro"
    # A tiny in-memory "database" the tool will read from.
    open_tickets: dict[str, list[str]]


agent = Agent(
    model,
    # Pass the TYPE, not an instance. The instance is supplied per-run via deps=.
    deps_type=CustomerCtx,
)


# Dynamic instructions: the system instruction is built per-run from ctx.deps.
@agent.instructions
def personalize(ctx: RunContext[CustomerCtx]) -> str:
    perk = "priority support" if ctx.deps.tier == "pro" else "standard support"
    return (
        f"You are a support assistant. The customer is {ctx.deps.name}, "
        f"on the {ctx.deps.tier} plan ({perk}). Address them by name and be brief."
    )


# A tool that reads injected deps via ctx.deps -- no globals, no args needed.
@agent.tool
def list_open_tickets(ctx: RunContext[CustomerCtx]) -> list[str]:
    """Return the current customer's open support tickets."""
    return ctx.deps.open_tickets.get(ctx.deps.name, [])


def main():
    tickets = {
        "Alice": ["#1042 login fails on mobile"],
        "Bob": [],
    }

    customers = [
        CustomerCtx(name="Alice", tier="pro", open_tickets=tickets),
        CustomerCtx(name="Bob", tier="free", open_tickets=tickets),
    ]

    question = "What's my support level, and do I have any open tickets?"

    for deps in customers:
        print("=" * 60)
        print(f"Injected deps -> name={deps.name!r} tier={deps.tier!r}")
        print("-" * 60)

        # Same agent, same prompt. Only `deps` differs between runs.
        result = agent.run_sync(question, deps=deps)
        print(f"Answer: {result.output}")

    print("=" * 60)


if __name__ == "__main__":
    main()
