"""Multi-agent orchestration with pydantic graphs.

Demonstrates: Graph-based agent orchestration, routing, dummy RAG pipeline.

Flow:
  Query -> Router Agent -> [RAG path] -> Retrieval -> Answer Agent -> Response
                       -> [Direct path] -> Direct Agent -> Response
"""

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from core import get_model

model = get_model()


# --- Structured outputs ---
class RouteType(str, Enum):
    RAG = "rag"  # Needs retrieval
    DIRECT = "direct"  # Can answer directly


class RouterOutput(BaseModel):
    route: RouteType
    reasoning: str


# --- Agents ---
router_agent = Agent(
    model,
    output_type=RouterOutput,
    instructions="""Classify if the query needs retrieval (rag) or can be answered directly (direct).

Our knowledge base contains info about: company details, product info, policies.

- rag: questions about company, product, policy, or specific organizational facts
- direct: general knowledge questions, explanations, opinions""",
)

answer_agent = Agent(
    model,
    instructions="Answer the question using ONLY the provided context. Be concise.",
)

direct_agent = Agent(
    model,
    instructions="Answer the question directly. Be concise and helpful.",
)


# --- Dummy vector store ---
DUMMY_DOCS = {
    "company": "Acme Corp was founded in 2020. CEO is Jane Smith. Revenue: $50M.",
    "product": "Widget X costs $99. Features: fast, reliable, blue color.",
    "policy": "Return policy: 30 days. Refunds processed in 5 business days.",
}


def dummy_retrieval(query: str) -> str:
    """Simulate vector store retrieval."""
    query_lower = query.lower()
    for key, doc in DUMMY_DOCS.items():
        if key in query_lower:
            return doc
    return "No relevant documents found."


# --- Graph state ---
@dataclass
class PipelineState:
    query: str = ""
    route: RouteType | None = None
    context: str = ""
    response: str = ""
    steps: list[str] = field(default_factory=list)


# --- Graph nodes ---
@dataclass
class RouterNode(BaseNode[PipelineState, None, str]):
    async def run(self, ctx: GraphRunContext[PipelineState]) -> "RetrievalNode | DirectNode":
        ctx.state.steps.append("router")
        result = await router_agent.run(ctx.state.query)
        ctx.state.route = result.output.route
        print(f"  [Router] {result.output.route.value}: {result.output.reasoning}")

        if result.output.route == RouteType.RAG:
            return RetrievalNode()
        return DirectNode()


@dataclass
class RetrievalNode(BaseNode[PipelineState, None, str]):
    async def run(self, ctx: GraphRunContext[PipelineState]) -> "AnswerNode":
        ctx.state.steps.append("retrieval")
        ctx.state.context = dummy_retrieval(ctx.state.query)
        print(f"  [Retrieval] Found: {ctx.state.context[:50]}...")
        return AnswerNode()


@dataclass
class AnswerNode(BaseNode[PipelineState, None, str]):
    async def run(self, ctx: GraphRunContext[PipelineState]) -> End[str]:
        ctx.state.steps.append("answer")
        prompt = f"Context: {ctx.state.context}\n\nQuestion: {ctx.state.query}"
        result = await answer_agent.run(prompt)
        ctx.state.response = result.output
        return End(result.output)


@dataclass
class DirectNode(BaseNode[PipelineState, None, str]):
    async def run(self, ctx: GraphRunContext[PipelineState]) -> End[str]:
        ctx.state.steps.append("direct")
        result = await direct_agent.run(ctx.state.query)
        ctx.state.response = result.output
        return End(result.output)


# --- Graph ---
graph = Graph(nodes=[RouterNode, RetrievalNode, AnswerNode, DirectNode])


def main():
    queries = [
        "What is the company revenue?",  # RAG path
        "What is the return policy?",  # RAG path
        "Explain what machine learning is",  # Direct path
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("-" * 60)

        state = PipelineState(query=query)
        result = graph.run_sync(RouterNode(), state=state)

        print(f"  [Response] {result.output}")
        print(f"  Path: {' -> '.join(state.steps)}")


if __name__ == "__main__":
    main()
