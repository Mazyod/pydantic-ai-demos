"""Human-in-the-loop: approval-required (deferred) tools.

Concept:
  Mark a sensitive tool with `requires_approval=True`. When the model wants to
  call it, Pydantic AI does NOT execute it. Instead, if `DeferredToolRequests`
  is in the agent's `output_type`, the run RETURNS early with
  `result.output` being a `DeferredToolRequests` listing the pending calls.

  A human then decides. You encode the decision in a `DeferredToolResults`
  (built via `requests.build_results(approvals={call_id: True/False})`) and
  RESUME the run by passing it back as `deferred_tool_results=`, along with
  `message_history=result.all_messages()`.

Goal / what to observe:
  Run 1 pauses with a pending `transfer_funds` approval (no model guessing,
  no money moved). We simulate a human approving it. Run 2 resumes, the tool
  finally executes, and the model produces a final confirmation - all with
  just ONE extra model round-trip after approval.
"""

from pydantic_ai import Agent, DeferredToolRequests, RunContext

from core import get_model

agent = Agent(
    get_model(),
    # Allowing both means: normal answers are `str`; a pending approval
    # surfaces as a `DeferredToolRequests` instead.
    output_type=[str, DeferredToolRequests],
    instructions=(
        "You are a banking assistant. To move money you MUST call "
        "transfer_funds. Do not ask the user to confirm in text - just call "
        "the tool; the system handles approval."
    ),
)


@agent.tool(requires_approval=True)
def transfer_funds(ctx: RunContext[None], to_account: str, amount_usd: float) -> str:
    """Transfer money to another account. Requires human approval."""
    # This body ONLY runs after a human approves (run 2).
    print(f"  [tool EXECUTED] transfer_funds(to={to_account}, ${amount_usd})")
    return f"Transfer of ${amount_usd:,.2f} to {to_account} completed. Ref #A1B2C3."


if __name__ == "__main__":
    print("=== Human-in-the-loop approval demo ===")
    print("Request: Transfer $250 to account ACME-9988.\n")

    # --- Run 1: model decides to call the tool; run pauses for approval ---
    print("[Run 1] Sending request to the model...")
    result = agent.run_sync("Please transfer $250 to account ACME-9988.")

    if not isinstance(result.output, DeferredToolRequests):
        print("\nModel did not request the protected tool. Output:")
        print(result.output)
        raise SystemExit(0)

    requests = result.output
    print("\n--- Run 1 PAUSED: approval required (tool NOT executed yet) ---")
    for call in requests.approvals:
        print(f"  pending: {call.tool_name}(args={call.args}) id={call.tool_call_id}")

    # --- Simulated human decision ---
    print("\n[Human] Reviewing the pending transfer... APPROVED.")
    decisions = {call.tool_call_id: True for call in requests.approvals}
    results = requests.build_results(approvals=decisions)

    # --- Run 2: resume with the approval; the tool now runs ---
    print("\n[Run 2] Resuming with the approval decision...")
    print("Tool activity:")
    final = agent.run_sync(
        message_history=result.all_messages(),
        deferred_tool_results=results,
    )

    print("\n--- Final answer (after approval + tool execution) ---")
    print(final.output)
