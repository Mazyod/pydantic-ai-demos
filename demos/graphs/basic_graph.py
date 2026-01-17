"""Basic graph demo - FSM without agents.

Demonstrates: State, Nodes, Graph, End, branching logic.
"""

from dataclasses import dataclass

from pydantic_graph import BaseNode, End, Graph, GraphRunContext


# State shared across all nodes
@dataclass
class CounterState:
    value: int = 0
    target: int = 10


# Node that increments the counter
@dataclass
class IncrementNode(BaseNode[CounterState, None, int]):
    amount: int = 1

    async def run(self, ctx: GraphRunContext[CounterState]) -> "CheckNode":
        ctx.state.value += self.amount
        print(f"  Incremented by {self.amount} -> {ctx.state.value}")
        return CheckNode()


# Node that checks if we've reached the target
@dataclass
class CheckNode(BaseNode[CounterState, None, int]):
    async def run(
        self, ctx: GraphRunContext[CounterState]
    ) -> "IncrementNode | End[int]":
        if ctx.state.value >= ctx.state.target:
            print(f"  Reached target! Final value: {ctx.state.value}")
            return End(ctx.state.value)

        # Increment by 2 if we're past halfway, else by 1
        amount = 2 if ctx.state.value > ctx.state.target // 2 else 1
        return IncrementNode(amount=amount)


# Create the graph
graph = Graph(nodes=[IncrementNode, CheckNode])


def main():
    print("Running graph...")
    state = CounterState(value=0, target=10)

    # Run synchronously
    result = graph.run_sync(IncrementNode(), state=state)

    print(f"\nResult: {result.output}")
    print(f"Final state: value={state.value}, target={state.target}")


if __name__ == "__main__":
    main()
