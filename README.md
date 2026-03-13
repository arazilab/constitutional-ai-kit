# Constitutional AI Package (CLI + GUI + Notebook)

This project is now a **Python package** with one shared constitutional AI engine.

- CLI calls the engine.
- GUI calls the same engine through a local HTTP API.
- Notebooks import and call the same engine directly.

No logic is duplicated between interfaces.

## What Changed

Before: single `index.html` file contained both UI and constitutional loop logic.

Now:

- `src/constitutional_ai/engine.py` contains the full writer/judge constitutional loop.
- `src/constitutional_ai/cli.py` provides command-line usage.
- `src/constitutional_ai/server.py` provides a local GUI server and API.
- `src/constitutional_ai/static/` contains GUI assets (client-only UI).
- `src/constitutional_ai/config.py` manages shared configuration.

## Install

From this repository root:

```bash
pip install -e .
```

Python requirement: `>=3.10`

## Use in Your Own Python Script

If someone clones this repo, they can use it as a library in any Python script after installing:

```bash
git clone <repo-url>
cd ConstitutionalAI_GUI
pip install -e .
```

Then in a script:

```python
from constitutional_ai.config import load_config
from constitutional_ai.engine import run_constitutional_turn
from constitutional_ai.models import ChatMessage

config = load_config()
thread = [ChatMessage(role="user", content="Summarize constitutional AI in 3 bullets.")]
turn = run_constitutional_turn(user_text=thread[-1].content, thread_messages=thread, config=config)
print(turn.final)
```

This uses the exact same engine and config model as the CLI and GUI.

## Quick Start

### 1. Set API key

```bash
export OPENAI_API_KEY="sk-..."
```

You can also save the key in the shared config file (see below), but env var is recommended.

### 2. Run CLI (single turn)

```bash
constitutional-ai run --prompt "What is constitutional AI?"
```

### 3. Run interactive CLI chat

```bash
constitutional-ai chat
```

### 4. Run GUI

```bash
constitutional-ai-gui
```

Then open `http://127.0.0.1:8765` if it does not open automatically.

## Shared Config (Used by CLI + GUI + Notebooks)

Default path:

- `~/.constitutional_ai/config.json`

Create it:

```bash
constitutional-ai config init
```

Inspect current effective config:

```bash
constitutional-ai config show --redact-key
```

### Config shape

```json
{
  "settings": {
    "api_key": "",
    "base_url": "https://api.openai.com",
    "writer_model": "gpt-4o-mini",
    "judge_model": "gpt-4o-mini",
    "temperature": 0.4,
    "max_tokens": 650,
    "max_revisions_per_rule": 1,
    "timeout_ms": 45000
  },
  "rules": ["..."],
  "prompts": {
    "writer_system": "...",
    "judge_pass_system": "...",
    "judge_critique_system": "..."
  }
}
```

The GUI reads/writes this same config via `/api/config`.

## Notebook Usage (Colab/Jupyter)

```python
from constitutional_ai.config import load_config
from constitutional_ai.engine import run_constitutional_turn
from constitutional_ai.models import ChatMessage

cfg = load_config()  # same config used by CLI/GUI
history = [ChatMessage(role="user", content="Explain gradient descent simply.")]
turn = run_constitutional_turn(user_text=history[-1].content, thread_messages=history, config=cfg)
print(turn.final)
```

This gives you the same constitution, prompts, and model settings as GUI/CLI.

## API Endpoints (Local GUI Server)

- `GET /` -> GUI
- `GET /api/config` -> current shared config
- `POST /api/config` -> merge and persist shared config
- `POST /api/turn` -> run one constitutional turn

## Development Notes

- `index.html` at repo root is now a simple pointer page.
- Main GUI assets live in `src/constitutional_ai/static/`.
- Package entrypoints:
  - `constitutional-ai`
  - `constitutional-ai-gui`

## Security Notes

- Prefer `OPENAI_API_KEY` via environment over storing API keys in files.
- If you store API keys in config, treat `~/.constitutional_ai/config.json` as sensitive.
- GUI traffic is local to your machine by default (`127.0.0.1`).
