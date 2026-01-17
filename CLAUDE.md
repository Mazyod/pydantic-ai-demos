# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This repository contains small, focused POCs and demos for learning Pydantic AI. Each example should demonstrate a single concept clearly without unnecessary complexity.

## Project Structure

```
pyd-ai/
├── src/core/          # Shared utilities (model config, common helpers)
├── demos/
│   ├── basics/        # Fundamental concepts
│   ├── tools/         # Tool usage demos
│   ├── streaming/     # Streaming responses
│   └── <category>/    # Add new categories as needed
│       └── <demo>/    # Multi-file demos get their own subfolder
```

## Commands

```bash
# Initial setup (creates venv, installs deps, installs project in editable mode)
uv sync

# Run a demo
uv run python demos/basics/hello.py

# Add dependencies
uv add <package>

# Sync dependencies
uv sync
```

## Guidelines

- **Keep POCs minimal** - Each demo should focus on one idea and drive it home
- **Avoid bloat** - No unnecessary abstractions, helpers, or over-engineering
- **Self-contained examples** - Each script should be runnable independently
- **Use Pydantic AI patterns** - Reference `/pydantic-ai` skill for framework guidance
- **Shared code** - Put reusable utilities in `src/core/`, import as `from core import ...`
- **Multi-file demos** - Create a subfolder within the category

## Pydantic AI Skill Feedback

If you encounter difficulties with the `/pydantic-ai` skill (missing information, outdated patterns, unclear examples), spawn a sub-agent to update the skill files at `~/.claude/skills/pydantic-ai/` with improvements based on what you learned. This helps enhance the skill over time.
