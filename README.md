# Constitutional AI Package (CLI + GUI + Notebook)

A Python package for constitutional AI with one shared engine across:

- CLI
- Local HTML GUI
- Python scripts and notebooks

This repo also includes `README.llm`, a structured machine-oriented companion doc for LLMs/coding agents.

## Install

From the repository root:

```bash
pip install -e .
```

Python requirement: `>=3.10`
Runtime dependency installed with package: `certifi` (for CA bundle handling).

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
constitutional-ai run --prompt "What is constitutional AI?" --show-metrics
```

Interactive chat:

```bash
constitutional-ai chat
constitutional-ai chat --execution-mode sequential
constitutional-ai chat --show-metrics
```

### 3. Run GUI

```bash
constitutional-ai-gui
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) if it does not auto-open.

On first run, the GUI auto-creates a starter config file and opens Settings if required setup (like API key) is missing.
In Settings, writer/judge model fields are dropdowns populated from live `/v1/models`.

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

Create starter config (optional, GUI auto-creates it if missing):

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
    "max_iteration_ms": 0,
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

`base_url` is normalized and validated (supports both `https://api.openai.com` and `https://api.openai.com/v1` as input).

For request-path debugging, set:

```bash
export CONSTITUTIONAL_AI_DEBUG=1
```

This logs final request URLs without exposing API keys.

TLS note:
- The shared client uses system certificates, supports `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE`, and will use `certifi` automatically when available.

Transcript note:
- Turn transcripts now include `run.events` timeline entries, including initial draft stage and sequential/parallel stage progress.
- Turn transcripts include `duration_ms`, so GUI/CLI/Python can all inspect elapsed turn time.
- `max_iteration_ms` can stop long runs and return the latest revision (`0` means no limit).

Model validation note:
- Before each turn, writer/judge model names are validated against the live model list.
- If a model is invalid, the error includes the available model IDs.

## Examples

See the [examples](./examples) folder for scenario-based examples with separate READMEs:

- different model choices
- multi-turn history handling
- different constitutions for different tasks
- parallel judge/critic execution mode with iteration cap controls
- single-turn minimal script usage

## LLM-Oriented Docs

- `README.llm` is a structured guide for coding agents (rules, signatures, examples).
- Keep `README.md` and `README.llm` updated together after each code edit so both reflect the latest behavior.
- Core invariant for contributors and agents: one shared backend core, multiple interfaces (CLI/GUI/scripts), no split backends.

### Prompt Example For LLMs

You can paste this template into an LLM prompt when requesting repo changes:

```text
Read `README.llm` in this repository and follow it strictly.

Task:
<describe the change>

Constraints:
1) Keep one shared backend core (`src/constitutional_ai/engine.py` + shared config/client layers).
2) Do not implement separate constitutional-processing logic for CLI vs GUI vs scripts.
3) If behavior changes, update both README.md and README.llm.
4) Reuse existing public APIs and provide minimal validation steps.
```

## Local API Endpoints (GUI server)

- `GET /` -> GUI
- `GET /api/config` -> current shared config
- `POST /api/config` -> merge and persist shared config
- `POST /api/test-connection` -> validate key/base URL/model connectivity from Settings
- `POST /api/models` -> list available models for GUI dropdowns
- `POST /api/turn-stream` -> run one turn and stream progress events (used by GUI status pill)
- `POST /api/turn-cancel` -> request cancellation of an active streamed turn
- `POST /api/turn` -> run one constitutional turn

## Security Notes

- Prefer `OPENAI_API_KEY` environment variable over storing keys in files.
- If you store API keys in config, treat `~/.constitutional_ai/config.json` as sensitive.
- GUI server binds to local host by default (`127.0.0.1`).
