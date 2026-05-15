"""Capability: `Hooks` — decorator lifecycle interception.

Concept
    `Hooks` is the ergonomic, no-subclass way to tap into the agent's ~20
    lifecycle interception points. You make a `Hooks()` instance, register
    functions via the `hooks.on.<event>` decorator namespace, and pass it in
    `capabilities=[hooks]`. Contract for the mutating hooks: receive the object,
    return it (optionally modified). This is the idiomatic replacement for
    hand-rolled agent wrappers / middleware.

    Decorator names live on the `hooks.on` namespace and are slightly shorter
    than the underlying `AbstractCapability` method names, e.g.:
        @hooks.on.before_run                -> AbstractCapability.before_run
        @hooks.on.before_model_request      -> AbstractCapability.before_model_request
        @hooks.on.after_model_request       -> AbstractCapability.after_model_request
        @hooks.on.after_run                 -> AbstractCapability.after_run

Goal
    Attach observe + mutate hooks around a single model call and watch them fire
    in order in stdout. One hook *modifies the response* so the effect is visible
    in the final output, not just in logs.

What to observe
    - The interleaved [hook] log lines proving each lifecycle point fired and in
      what order (before_run -> before_model_request -> after_model_request ->
      after_run), with no agent-subclassing.
    - `before_model_request` reports how many messages are being sent.
    - `after_model_request` rewrites the model's text, so the WITH-hooks output
      carries a banner the model never produced — the WITHOUT run is shown first
      for contrast.
"""

from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelResponse, TextPart

from core import get_model

PROMPT = "In one short sentence, what is a hook in software?"


def build_hooks() -> Hooks:
    hooks = Hooks()

    @hooks.on.before_run
    async def _before_run(ctx: RunContext[None]) -> None:
        # Observe-only hook (returns None). Fires once, before anything else.
        name = ctx.agent.name if ctx.agent else "unnamed"
        print(f"[hook] before_run        : agent={name!r} run starting")

    @hooks.on.before_model_request
    async def _before_model_request(
        ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        # Mutating hook: must return a ModelRequestContext (here unchanged).
        n = len(request_context.messages)
        print(f"[hook] before_model_req  : sending {n} message(s) to the model")
        return request_context

    @hooks.on.after_model_request
    async def _after_model_request(
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        # Mutating hook: rewrite the model's text so the change is visible
        # downstream in result.output.
        print("[hook] after_model_req   : rewriting model response text")
        new_parts = []
        for part in response.parts:
            if isinstance(part, TextPart):
                new_parts.append(TextPart(content=f"[via after_model_request hook] {part.content}"))
            else:
                new_parts.append(part)
        return ModelResponse(parts=new_parts, model_name=response.model_name)

    @hooks.on.after_run
    async def _after_run(ctx: RunContext[None], *, result):
        # Mutating hook: must return the (possibly modified) AgentRunResult.
        print(f"[hook] after_run         : run finished, output len={len(result.output)}")
        return result

    return hooks


def run(label: str, *capabilities) -> None:
    print(f"=== {label} ===")
    agent = Agent(get_model(), name="hooked", instructions="Be concise.", capabilities=list(capabilities))
    result = agent.run_sync(PROMPT)
    print(f"final output: {result.output.strip()}")
    print()


if __name__ == "__main__":
    run("WITHOUT hooks (plain Agent)")
    run("WITH Hooks() capability", build_hooks())
    print(
        "Takeaway: a `Hooks()` instance in capabilities=[...] intercepts the\n"
        "lifecycle without subclassing. Observe-only hooks return None; mutating\n"
        "hooks receive an object and return it (optionally changed)."
    )
