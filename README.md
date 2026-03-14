# Constitutional AI Package (CLI + GUI + Notebook)

A Python package for constitutional AI with one shared engine across:

- CLI
- Local HTML GUI
- Python scripts and notebooks

## Install

From the repository root:

```bash
pip install -e .
```

Python requirement: `>=3.10`

## Quick Start

### 1. Set API key

```bash
export OPENAI_API_KEY="sk-..."
```

### 2. Run CLI

Single turn:

```bash
constitutional-ai run --prompt "What is constitutional AI?"
constitutional-ai run --prompt "What is constitutional AI?" --execution-mode parallel --parallel-max-iterations 1
```

Interactive chat:

```bash
constitutional-ai chat
constitutional-ai chat --execution-mode sequential
```

### 3. Run GUI

```bash
constitutional-ai-gui
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) if it does not auto-open.

## Use in Your Own Python Script

After cloning and installing this repo, any Python script can import the package:

```python
from constitutional_ai.config import load_config
from constitutional_ai.engine import run_constitutional_turn
from constitutional_ai.models import ChatMessage

config = load_config()
thread = [ChatMessage(role="user", content="Summarize constitutional AI in 3 bullets.")]
turn = run_constitutional_turn(user_text=thread[-1].content, thread_messages=thread, config=config)
print(turn.final)
```

This uses the same engine and config model as the CLI and GUI.

## Shared Config

Default path:

- `~/.constitutional_ai/config.json`

Create starter config:

```bash
constitutional-ai config init
```

Show effective config:

```bash
constitutional-ai config show --redact-key
```

Config shape:

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
    "execution_mode": "sequential",
    "parallel_max_iterations": 0,
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

The GUI reads and writes this same config via `/api/config`.

## Examples

See the [examples](./examples) folder for scenario-based examples with separate READMEs:

- different model choices
- multi-turn history handling
- different constitutions for different tasks
- single-turn minimal script usage

## Local API Endpoints (GUI server)

- `GET /` -> GUI
- `GET /api/config` -> current shared config
- `POST /api/config` -> merge and persist shared config
- `POST /api/turn` -> run one constitutional turn

## Security Notes

- Prefer `OPENAI_API_KEY` environment variable over storing keys in files.
- If you store API keys in config, treat `~/.constitutional_ai/config.json` as sensitive.
- GUI server binds to local host by default (`127.0.0.1`).
