"""FunctionModel: script the model's behavior for deterministic tests.

Concept: where `TestModel` auto-drives tools generically, `FunctionModel` lets
you supply a plain function that returns the exact `ModelResponse` for each
step. You decide turn-by-turn whether the model emits a tool call or final
text, so you can simulate a precise multi-step conversation with NO network.

Goal: take an ordinary agent, swap its model for a scripted `FunctionModel`
via `agent.override(model=...)` (the idiomatic test pattern), and show the
deterministic, reproducible result of a two-step run: model calls a tool, then
uses the tool's return to produce the final answer.
"""

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

agent = Agent(
    "openai-chat:gpt-5.2",  # never contacted: overridden with FunctionModel
    instructions="Convert currency for the user.",
)


@agent.tool_plain
def usd_to_kwd(amount: float) -> float:
    """Convert USD to KWD at a fixed demo rate."""
    return round(amount * 0.31, 2)


def scripted_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Decide the response based on how far the conversation has progressed.

    Step 1 (only a user prompt so far) -> call the `usd_to_kwd` tool.
    Step 2 (a tool result is now present) -> emit the final text answer.
    """
    tool_returns = [
        part
        for m in messages
        if isinstance(m, ModelRequest)
        for part in m.parts
        if isinstance(part, ToolReturnPart)
    ]

    if not tool_returns:
        # First step: instruct the agent to call our tool with fixed args.
        return ModelResponse(
            parts=[ToolCallPart("usd_to_kwd", {"amount": 100.0})]
        )

    # Second step: the tool ran; weave its result into a final answer.
    converted = tool_returns[-1].content
    return ModelResponse(
        parts=[TextPart(f"100 USD is {converted} KWD.")]
    )


if __name__ == "__main__":
    print("=== FunctionModel scripted run (no network) ===")

    # `agent.override` swaps the model for the whole `with` block -- exactly how
    # you'd wrap a real agent inside a test without touching its definition.
    with agent.override(model=FunctionModel(scripted_model)):
        result = agent.run_sync("Convert 100 dollars to dinar.")

    print("Final output :", result.output)
    print("Model name   :", result.all_messages()[-1].model_name)

    # Deterministic: rerun and confirm byte-identical output.
    with agent.override(model=FunctionModel(scripted_model)):
        again = agent.run_sync("Convert 100 dollars to dinar.")

    print("Rerun output :", again.output)
    assert result.output == again.output == "100 USD is 31.0 KWD."
    print("Assertion    : both runs produced the identical scripted result")
