"""Deterministic multi-agent routing demo.

Demonstrates: Multiple agents, structured output for routing, explicit code-controlled transitions.

Flow:
  User query -> Router (classifies) -> Code decides -> Specialist agent -> Response
"""

from enum import Enum

from pydantic import BaseModel
from pydantic_ai import Agent

from core import get_model

model = get_model()


# Structured output for deterministic routing
class QueryType(str, Enum):
    TECHNICAL = "technical"
    GENERAL = "general"


class RouterOutput(BaseModel):
    query_type: QueryType
    reasoning: str


# Router agent - classifies the query
router = Agent(
    model,
    output_type=RouterOutput,
    instructions="""Classify the user's query as either:
- technical: programming, code, APIs, debugging, software
- general: everything else

Be concise in your reasoning.""",
)

# Specialist agents
tech_agent = Agent(
    model,
    instructions="You are a technical expert. Give concise, accurate technical answers.",
)

general_agent = Agent(
    model,
    instructions="You are a friendly general assistant. Be helpful and conversational.",
)


def main():
    queries = [
        "How do I reverse a list in Python?",
        "What's a good recipe for pasta?",
        "Explain the difference between async and sync",
    ]

    for query in queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print("-" * 50)

        # Step 1: Route the query
        route_result = router.run_sync(query)
        classification = route_result.output
        print(f"Router: {classification.query_type.value} ({classification.reasoning})")

        # Step 2: Deterministic dispatch based on classification
        if classification.query_type == QueryType.TECHNICAL:
            agent = tech_agent
        else:
            agent = general_agent

        # Step 3: Get specialist response
        response = agent.run_sync(query)
        print(f"Response: {response.output}")


if __name__ == "__main__":
    main()
