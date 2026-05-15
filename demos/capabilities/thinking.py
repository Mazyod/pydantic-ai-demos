"""Capability: `Thinking` — portable reasoning-effort control.

Concept
    `capabilities=[...]` is the single modern configuration surface for an agent
    (introduced v1.71). `Thinking` is the cross-provider capability that turns on
    the model's reasoning/thinking and sets its effort level. Under the hood it
    contributes nothing but `ModelSettings(thinking=<effort>)` — the *unified*
    thinking setting — which each provider maps to its own native knob (OpenAI
    reasoning effort, Anthropic thinking budget, Qwen `/think`, etc.). That is the
    whole point of the capability: one portable surface instead of N provider
    kwargs.

Goal
    Run the same reasoning task twice — once plain, once with
    `Thinking(effort="high")` — and (a) prove the capability deterministically
    injects the unified `thinking` setting, and (b) inspect the response for the
    model's `ThinkingPart` reasoning trace.

What to observe
    - The capability's deterministic effect: `Thinking(effort='high')` resolves to
      `{'thinking': 'high'}` in model settings (printed). This is true on every
      provider, which is the portability win.
    - `ThinkingPart`s in `result.all_messages()`: the model's reasoning trace,
      kept *separate* from the final answer in `result.output`.
    - Caveat (visible here): the local Qwen3 model is an *always-on* reasoning
      model, so it emits a ThinkingPart even WITHOUT the capability — `Thinking`
      is then "silently ignored" per its own docstring. The lesson on such models
      is the portable settings surface, not an on/off toggle. On a model where
      thinking is opt-in (e.g. OpenAI o-series), the WITHOUT run would have zero
      thinking parts and the WITH run would have them.
"""

from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking
from pydantic_ai.messages import ThinkingPart

from core import get_model

QUESTION = (
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the "
    "ball. How much does the ball cost? Answer with just the amount."
)


def thinking_parts(result) -> list[str]:
    """Pull every ThinkingPart's content out of a run result's messages."""
    parts: list[str] = []
    for message in result.all_messages():
        for part in getattr(message, "parts", []):
            if isinstance(part, ThinkingPart) and part.has_content():
                parts.append(part.content)
    return parts


def run(label: str, *capabilities) -> None:
    agent = Agent(get_model(), instructions="Be concise.", capabilities=list(capabilities))
    result = agent.run_sync(QUESTION)
    thoughts = thinking_parts(result)

    print(f"=== {label} ===")
    if capabilities:
        cap = capabilities[0]
        # The deterministic, provider-portable contribution of the capability:
        print(f"capability          : {cap!r}")
        print(f"-> model settings   : {cap.get_model_settings()}")
    else:
        print("capability          : (none — plain Agent)")
    print(f"thinking parts       : {len(thoughts)}")
    if thoughts:
        chars = len(thoughts[0])
        preview = thoughts[0].strip().replace("\n", " ")
        print(f"first thought ({chars:>4} chars): {preview[:150]}...")
    print(f"final output         : {result.output.strip()}")
    print()


if __name__ == "__main__":
    print("Same reasoning question, two configs.")
    print("Watch the capability inject the portable `thinking` setting.\n")
    run("WITHOUT Thinking capability")
    run("WITH Thinking(effort='high')", Thinking(effort="high"))
    print(
        "Takeaway: `capabilities=[Thinking(effort=...)]` is the one portable\n"
        "reasoning surface. It maps to ModelSettings(thinking=...) for ANY\n"
        "provider; explicit provider kwargs still win if you set both."
    )
