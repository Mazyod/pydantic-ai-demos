# Pydantic AI Demos

A collection of minimal, focused demos for learning [Pydantic AI](https://ai.pydantic.dev/).

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Run a demo
uv run python demos/basics/hello.py
```

## Project Structure

```
demos/
├── basics/          # Fundamental concepts
│   └── hello.py     # Simple agent demo
└── graphs/          # Pydantic Graph demos
    └── basic_graph.py  # FSM-style graph without agents

src/core/            # Shared utilities
└── models.py        # Model configuration (LM Studio compatible)
```

## Demos

### basics/hello.py
Simple agent that answers questions. Demonstrates basic `Agent` usage with `run_sync`.

### graphs/basic_graph.py
Counter FSM using Pydantic Graph. Demonstrates `BaseNode`, `End`, `Graph`, state management, and branching logic.

## Configuration

By default, demos use LM Studio at `localhost:1234`. Edit `src/core/models.py` to change the model or endpoint.

## License

MIT
