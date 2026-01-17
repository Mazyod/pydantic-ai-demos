"""Agent interface demo - expose agent as CLI or Web.

Demonstrates: Agent creation, to_cli(), to_web(), rich prompts.
"""

import asyncio

import uvicorn
from pydantic_ai import Agent
from rich.prompt import Prompt

from core import get_model

agent = Agent(
    get_model(),
    instructions="You are a helpful assistant. Be concise and friendly.",
)


async def main():
    choice = Prompt.ask(
        "How would you like to interact with the agent?",
        choices=["cli", "web"],
        default="cli",
    )

    if choice == "cli":
        await agent.to_cli()
    else:
        app = agent.to_web()
        config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
        server = uvicorn.Server(config)
        try:
            await server.serve()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
