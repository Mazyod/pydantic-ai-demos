"""Swapping dependencies with agent.override(deps=...).

Demonstrates: `agent.override(deps=...)` -- a context manager that forces the
agent to use alternate dependencies for everything inside the `with` block,
WITHOUT changing the agent or how it's called. Ideal for tests or pointing the
same agent at a fake/alternate environment.

Goal: define one agent with a "production" data source, then prove that
override swaps in a fake source for a region of code and automatically
restores the original afterwards.
"""

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from core import get_model

model = get_model()


@dataclass
class WeatherDeps:
    # A label so the printed output makes the active dependency obvious.
    source_name: str
    # Pretend this is a live API client; here it's just a lookup function.
    get_temp_c: callable


agent = Agent(model, deps_type=WeatherDeps)


@agent.instructions
def which_source(ctx: RunContext[WeatherDeps]) -> str:
    return (
        f"You report temperatures using the {ctx.deps.source_name} data source. "
        f"Always state the source name. Be brief."
    )


@agent.tool
def temperature_c(ctx: RunContext[WeatherDeps], city: str) -> str:
    """Return the current temperature for a city, in Celsius."""
    return f"{ctx.deps.get_temp_c(city)}C (via {ctx.deps.source_name})"


# "Production" deps: would normally hit a real service.
prod_deps = WeatherDeps(
    source_name="LIVE-API",
    get_temp_c=lambda city: 31,
)

# "Test" deps: deterministic fake, no network. Same shape as prod_deps.
fake_deps = WeatherDeps(
    source_name="FAKE-FIXTURE",
    get_temp_c=lambda city: 99,
)


def main():
    question = "What's the temperature in Kuwait City?"

    print("=" * 60)
    print("1) Normal run -- uses the deps passed to run_sync (LIVE-API):")
    print("-" * 60)
    r1 = agent.run_sync(question, deps=prod_deps)
    print(f"Answer: {r1.output}")

    print("=" * 60)
    print("2) Inside agent.override(deps=fake_deps) -- deps are forced to the")
    print("   fake fixture even though we still pass prod_deps to run_sync:")
    print("-" * 60)
    with agent.override(deps=fake_deps):
        r2 = agent.run_sync(question, deps=prod_deps)
    print(f"Answer: {r2.output}")

    print("=" * 60)
    print("3) After the with-block -- override is gone, back to LIVE-API:")
    print("-" * 60)
    r3 = agent.run_sync(question, deps=prod_deps)
    print(f"Answer: {r3.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
