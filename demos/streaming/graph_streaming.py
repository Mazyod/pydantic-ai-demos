"""Graph streaming demo - UIEventStream with multi-agent orchestration.

Demonstrates:
- Using UIEventStream directly (not UIAdapter) with graphs
- Streaming events from nested agents through graph nodes
- Plain SSE output for CLI/web consumption

Key insight from Pydantic AI maintainer (Issue #3884):
> "UIEventStream exists exactly for the scenario where you have a Pydantic AI
>  event stream rather than a specific agent to run."

Run CLI mode:
    uv run python demos/streaming/graph_streaming.py --cli "What is the company revenue?"

Run server mode:
    uv run python demos/streaming/graph_streaming.py --serve
    # Then: curl -N -X POST http://localhost:8000/chat \
    #   -H "Content-Type: application/json" \
    #   -d '{"query": "What is the return policy?"}'
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import (
    AgentStreamEvent,
    TextPartDelta,
)
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.ui import UIEventStream
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from core import get_model

# Type alias for native events (same as in pydantic_ai.ui._event_stream)
NativeEvent = AgentStreamEvent | AgentRunResultEvent[Any]


# --- Model setup ---
model = get_model()


# --- Structured outputs ---
class RouteType(str, Enum):
    RAG = "rag"
    DIRECT = "direct"


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


# --- SSE Event type ---
@dataclass
class SSEEvent:
    """Simple SSE event."""

    event: str
    data: dict[str, Any]


# --- Simple SSE Event Stream ---
@dataclass
class SimpleSSEEventStream(UIEventStream[None, SSEEvent, None, str]):
    """Minimal SSE event stream for graph streaming demo.

    Transforms Pydantic AI native events into simple SSE events.
    """

    def encode_event(self, event: SSEEvent) -> str:
        """Encode SSE event as string."""
        return f"event: {event.event}\ndata: {json.dumps(event.data)}\n\n"

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[SSEEvent]:
        """Handle text deltas - the main streaming content."""
        if delta.content_delta:
            yield SSEEvent("text_delta", {"content": delta.content_delta})

    async def handle_text_start(self, part, follows_text: bool = False) -> AsyncIterator[SSEEvent]:
        """Handle text part start."""
        if not follows_text:
            yield SSEEvent("text_start", {})

    async def handle_text_end(self, part, followed_by_text: bool = False) -> AsyncIterator[SSEEvent]:
        """Handle text part end."""
        if not followed_by_text:
            yield SSEEvent("text_end", {})

    async def handle_run_result(self, event: AgentRunResultEvent) -> AsyncIterator[SSEEvent]:
        """Handle agent run completion."""
        yield SSEEvent("agent_done", {"output": str(event.result.output)})

    async def before_stream(self) -> AsyncIterator[SSEEvent]:
        """Emit event at stream start."""
        yield SSEEvent("stream_start", {})

    async def after_stream(self) -> AsyncIterator[SSEEvent]:
        """Emit event at stream end."""
        yield SSEEvent("stream_end", {})


# --- Custom event for graph node transitions ---
@dataclass
class NodeEvent:
    """Custom event for graph node transitions."""

    node_name: str
    event_type: str  # "start" or "end"


# --- Streaming-aware graph nodes ---
@dataclass
class StreamingRouterNode(BaseNode[PipelineState, None, str]):
    """Router node that streams events."""

    send_stream: MemoryObjectSendStream[NativeEvent | NodeEvent]

    async def run(
        self, ctx: GraphRunContext[PipelineState]
    ) -> "StreamingRetrievalNode | StreamingDirectNode":
        ctx.state.steps.append("router")

        # Emit node start
        await self.send_stream.send(NodeEvent("router", "start"))

        # Stream events from router agent
        async for event in router_agent.run_stream_events(ctx.state.query):
            await self.send_stream.send(event)

            # Extract result from final event
            if isinstance(event, AgentRunResultEvent):
                ctx.state.route = event.result.output.route

        await self.send_stream.send(NodeEvent("router", "end"))

        if ctx.state.route == RouteType.RAG:
            return StreamingRetrievalNode(send_stream=self.send_stream)
        return StreamingDirectNode(send_stream=self.send_stream)


@dataclass
class StreamingRetrievalNode(BaseNode[PipelineState, None, str]):
    """Retrieval node (no agent, just fetches context)."""

    send_stream: MemoryObjectSendStream[NativeEvent | NodeEvent]

    async def run(self, ctx: GraphRunContext[PipelineState]) -> "StreamingAnswerNode":
        ctx.state.steps.append("retrieval")

        await self.send_stream.send(NodeEvent("retrieval", "start"))
        ctx.state.context = dummy_retrieval(ctx.state.query)
        await self.send_stream.send(NodeEvent("retrieval", "end"))

        return StreamingAnswerNode(send_stream=self.send_stream)


@dataclass
class StreamingAnswerNode(BaseNode[PipelineState, None, str]):
    """Answer node that streams response using RAG context."""

    send_stream: MemoryObjectSendStream[NativeEvent | NodeEvent]

    async def run(self, ctx: GraphRunContext[PipelineState]) -> End[str]:
        ctx.state.steps.append("answer")

        await self.send_stream.send(NodeEvent("answer", "start"))

        prompt = f"Context: {ctx.state.context}\n\nQuestion: {ctx.state.query}"

        async for event in answer_agent.run_stream_events(prompt):
            await self.send_stream.send(event)

            if isinstance(event, AgentRunResultEvent):
                ctx.state.response = event.result.output

        await self.send_stream.send(NodeEvent("answer", "end"))

        return End(ctx.state.response)


@dataclass
class StreamingDirectNode(BaseNode[PipelineState, None, str]):
    """Direct answer node that streams response."""

    send_stream: MemoryObjectSendStream[NativeEvent | NodeEvent]

    async def run(self, ctx: GraphRunContext[PipelineState]) -> End[str]:
        ctx.state.steps.append("direct")

        await self.send_stream.send(NodeEvent("direct", "start"))

        async for event in direct_agent.run_stream_events(ctx.state.query):
            await self.send_stream.send(event)

            if isinstance(event, AgentRunResultEvent):
                ctx.state.response = event.result.output

        await self.send_stream.send(NodeEvent("direct", "end"))

        return End(ctx.state.response)


# --- Graph definition ---
graph = Graph(
    nodes=[
        StreamingRouterNode,
        StreamingRetrievalNode,
        StreamingAnswerNode,
        StreamingDirectNode,
    ]
)


# --- Event aggregator ---
async def run_graph_with_streaming(query: str) -> AsyncIterator[NativeEvent | NodeEvent]:
    """Run graph and yield all events from nested agents.

    This is the key function that aggregates events from all graph nodes
    into a single async iterator that can be passed to UIEventStream.transform_stream().
    """
    send_stream, receive_stream = anyio.create_memory_object_stream[NativeEvent | NodeEvent](32)
    state = PipelineState(query=query)

    async def run_graph():
        async with send_stream:
            initial_node = StreamingRouterNode(send_stream=send_stream)
            await graph.run(initial_node, state=state)

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_graph)
        async with receive_stream:
            async for event in receive_stream:
                yield event


# --- Extended event stream that handles NodeEvent ---
@dataclass
class GraphSSEEventStream(SimpleSSEEventStream):
    """SSE event stream that also handles graph node events."""

    async def transform_stream(
        self,
        stream: AsyncIterator[NativeEvent | NodeEvent],
        on_complete=None,
    ) -> AsyncIterator[SSEEvent]:
        """Transform stream, handling both native events and node events."""
        async for e in self.before_stream():
            yield e

        try:
            async for event in stream:
                # Handle custom NodeEvent
                if isinstance(event, NodeEvent):
                    yield SSEEvent(
                        f"node_{event.event_type}",
                        {"node": event.node_name},
                    )
                else:
                    # Delegate to parent for native events
                    async for e in self.handle_event(event):
                        yield e
        finally:
            async for e in self.after_stream():
                yield e


# --- CLI mode ---
async def run_cli(query: str) -> None:
    """Run in CLI mode - print events as they stream."""
    print(f"\nQuery: {query}")
    print("-" * 60)

    event_stream = GraphSSEEventStream(run_input=None)
    native_events = run_graph_with_streaming(query)

    async for event in event_stream.transform_stream(native_events):
        if event.event == "text_delta":
            print(event.data["content"], end="", flush=True)
        elif event.event == "node_start":
            print(f"\n[{event.data['node']}] starting...")
        elif event.event == "node_end":
            print(f"[{event.data['node']}] done")
        elif event.event == "stream_end":
            print("\n" + "-" * 60)
            print("Stream complete.")


# --- Server mode ---
def create_app():
    """Create FastAPI app for server mode."""
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel as PydanticBaseModel
    except ImportError:
        raise ImportError(
            "FastAPI is required for server mode. Install with: uv add fastapi"
        )

    app = FastAPI(title="Graph Streaming Demo")

    class ChatRequest(PydanticBaseModel):
        query: str

    @app.post("/chat")
    async def chat(request: ChatRequest):
        """Stream chat response from graph-based multi-agent pipeline."""
        event_stream = GraphSSEEventStream(run_input=None)
        native_events = run_graph_with_streaming(request.query)
        protocol_events = event_stream.transform_stream(native_events)
        return event_stream.streaming_response(protocol_events)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required for server mode. Install with: uv add uvicorn"
        )

    app = create_app()
    uvicorn.run(app, host=host, port=port)


# --- Main ---
def main():
    parser = argparse.ArgumentParser(
        description="Graph streaming demo with UIEventStream"
    )
    parser.add_argument(
        "--cli",
        type=str,
        metavar="QUERY",
        help="Run in CLI mode with the given query",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run in server mode (FastAPI)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for server mode (default: 8000)",
    )

    args = parser.parse_args()

    if args.cli:
        asyncio.run(run_cli(args.cli))
    elif args.serve:
        run_server(port=args.port)
    else:
        # Default: run a demo query in CLI mode
        asyncio.run(run_cli("What is the company revenue?"))


if __name__ == "__main__":
    main()
