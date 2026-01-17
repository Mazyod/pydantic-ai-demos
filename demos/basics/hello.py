"""Basic hello world demo with Pydantic AI."""

from pydantic_ai import Agent

from core import get_model

agent = Agent(get_model(), instructions="Be concise.")

result = agent.run_sync("What is 2 + 2?")
print(result.output)
