"""TestModel: run an agent with NO API call.

Concept: `pydantic_ai.models.test.TestModel` is a fake model that, without any
network access, auto-drives every function tool the agent exposes and then
synthesizes a value for the agent's `output_type`. It is the cheapest way to
exercise an agent's wiring (tools called, structured output shape) in a unit
test.

Goal: define a realistic agent (a tool + a structured `output_type`), run it
under `TestModel`, and print exactly what TestModel did -- entirely offline and
deterministic.
"""

from pydantic import BaseModel

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel


class WeatherReport(BaseModel):
    """Structured output the agent must produce."""

    city: str
    summary: str
    temperature_c: int


# A normal agent. In production this would point at a live model via get_model().
agent = Agent(
    "openai-chat:gpt-5.2",  # never contacted: TestModel is injected below
    output_type=WeatherReport,
    instructions="Report the weather for the requested city.",
)


@agent.tool_plain
def get_temperature(city: str) -> int:
    """Look up the current temperature for a city (stub for the demo)."""
    return {"Kuwait City": 38, "London": 14}.get(city, 20)


if __name__ == "__main__":
    # 1) TestModel auto-mode: calls every tool, then fills the output model.
    with capture_run_messages() as messages:
        result = agent.run_sync("Weather in Kuwait City?", model=TestModel())

    print("=== 1. TestModel auto-driven run (no network) ===")
    print("Structured output :", result.output)
    print("Output type       :", type(result.output).__name__)

    tool_calls = [
        p.tool_name
        for m in messages
        for p in m.parts
        if isinstance(p, ToolCallPart)
    ]
    print("Tool calls issued :", tool_calls)
    print("Usage             :", result.usage)

    # 2) TestModel can be pinned to deterministic output args, so a test can
    #    assert on an exact value instead of TestModel's generated placeholder.
    pinned = TestModel(
        custom_output_args={
            "city": "London",
            "summary": "Light rain, overcast.",
            "temperature_c": 14,
        }
    )
    result2 = agent.run_sync("Weather in London?", model=pinned)

    print()
    print("=== 2. TestModel with pinned custom_output_args ===")
    print("Structured output :", result2.output)
    assert result2.output.city == "London", "pinned output should be exact"
    assert result2.output.temperature_c == 14
    print("Assertions passed : output matched the pinned values exactly")
