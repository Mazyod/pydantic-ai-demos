"""Capability: custom `AbstractCapability` — a reusable output guardrail.

Concept
    For application code that just needs a few hooks, use `Hooks()`. For a
    *reusable, composable* unit of agent behavior, subclass `AbstractCapability`.
    A capability can contribute static config (`get_instructions`,
    `get_model_settings`, `get_toolset`, `get_native_tools`) AND override
    lifecycle hook methods (`before_model_request`, `after_model_request`, ...).
    It is a `@dataclass`, parameterized by its fields, and dropped into
    `capabilities=[...]` like any built-in.

    This demo builds `PiiRedactor`, which does two things at once:
      * `get_instructions()`     — injects a system instruction (static config).
      * `after_model_request()`  — redacts emails/phones from the model's text as
        a belt-and-suspenders guardrail (lifecycle hook).
    Both compose with the built-in `Thinking` capability in the same list to show
    capabilities stack.

    Hook choice note (a real gotcha): the output-validation hooks
    (`after_output_validate`) only fire for *structured* output that needs
    parsing — NOT for plain text (confirmed in the installed
    `AbstractCapability` source). For a plain-text guardrail the right hook is
    `after_model_request`, which sees every `ModelResponse` and can rewrite its
    parts before they ever become `result.output`.

Goal
    Show one custom capability simultaneously shaping the prompt and
    post-processing the model's text, and visibly changing `result.output`.

What to observe
    - WITHOUT the capability: the model emits a synthetic email/phone verbatim.
    - WITH `PiiRedactor()`: the injected instruction nudges the model, AND the
      `after_model_request` hook hard-redacts anything that still leaks, so the
      final output contains `[REDACTED-EMAIL]` / `[REDACTED-PHONE]` markers
      regardless of what the model did. The redaction-hit counter is printed,
      proving the *hook* fired (a guardrail must not trust the model to comply).
    - It composes alongside `Thinking(effort='low')` in `capabilities=[...]`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability, Thinking
from pydantic_ai.messages import ModelResponse, TextPart

from core import get_model

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\(?\d[\d\s().-]{7,}\d")

PROMPT = (
    "Generate ONE line of fake sample test data for a contact form: a made-up "
    "name, a made-up email, and a made-up phone number. Output only that line."
)


@dataclass
class PiiRedactor(AbstractCapability[Any]):
    """Reusable guardrail: steer the model away from PII, then redact any leak.

    Fields make the capability configurable, exactly like the built-ins
    (`Thinking(effort=...)`). State here (`hits`) demonstrates a capability can
    also accumulate run info.
    """

    email_token: str = "[REDACTED-EMAIL]"
    phone_token: str = "[REDACTED-PHONE]"
    hits: list[str] = field(default_factory=list)

    def get_instructions(self) -> str:
        # Static config contribution, collected once at agent construction.
        # A soft steer — note the hook below does NOT trust the model to obey it.
        return "Prefer to keep personal contact details out of your answers."

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        # Lifecycle hook: fires after every model response, before it becomes
        # output. Rewrite TextParts in place; return the modified ModelResponse.
        new_parts = []
        for part in response.parts:
            if isinstance(part, TextPart):
                text, n_email = EMAIL_RE.subn(self.email_token, part.content)
                text, n_phone = PHONE_RE.subn(self.phone_token, text)
                if n_email or n_phone:
                    self.hits.append(f"{n_email} email(s), {n_phone} phone(s)")
                new_parts.append(TextPart(content=text))
            else:
                new_parts.append(part)
        return ModelResponse(parts=new_parts, model_name=response.model_name)


def run(label: str, *capabilities) -> None:
    print(f"=== {label} ===")
    agent = Agent(get_model(), instructions="Be concise.", capabilities=list(capabilities))
    result = agent.run_sync(PROMPT)
    print(f"capabilities : {[type(c).__name__ for c in capabilities] or '(none)'}")
    print(f"final output : {result.output.strip()}")
    for cap in capabilities:
        if isinstance(cap, PiiRedactor):
            print(f"redactions   : {cap.hits or 'none'}")
    print()


if __name__ == "__main__":
    run("WITHOUT custom capability")
    # The custom capability composes with a built-in one in the same list.
    run("WITH PiiRedactor() + Thinking(effort='low')", PiiRedactor(), Thinking(effort="low"))
    print(
        "Takeaway: subclass AbstractCapability for a reusable unit that can both\n"
        "contribute static config (get_instructions) and intercept the lifecycle\n"
        "(after_model_request). It drops into capabilities=[...] and composes\n"
        "with built-ins like Thinking. Pick the hook that fires for your output\n"
        "type: after_model_request for plain text, *_output_validate for\n"
        "structured output."
    )
