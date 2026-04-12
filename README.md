# Constitutional AI Package

One shared constitutional AI engine with three interfaces:

- CLI
- local GUI
- Python scripts / notebooks

The model layer now uses [LiteLLM](https://github.com/BerriAI/litellm), with OpenAI as the default configuration and support for other providers such as Anthropic, Gemini, xAI, OpenRouter, Groq, Together AI, Hugging Face, Azure OpenAI, and local providers like Ollama or LM Studio.

## Install

From the repository root:

```bash
pip install -e .
```

Python requirement: `>=3.10`

## Quick start

### 1. Configure credentials

You can use either environment variables or the shared config / GUI.

OpenAI example:

```bash
export OPENAI_API_KEY="sk-..."
```

Anthropic example:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Gemini example:

```bash
export GEMINI_API_KEY="AIza..."
```

### 2. Run the CLI

Single turn:

```bash
constitutional-ai run --prompt "What is constitutional AI?"
constitutional-ai run --prompt "What is constitutional AI?" --execution-mode parallel --parallel-max-iterations 1
constitutional-ai run --prompt "What is constitutional AI?" --show-metrics
```

Use a non-default provider:

```bash
constitutional-ai run \
  --prompt "Summarize constitutional AI" \
  --writer-provider anthropic \
  --writer-model claude-sonnet-4-5-20250929 \
  --judge-provider openai \
  --judge-model gpt-4o-mini \
  --anthropic-api-key "$ANTHROPIC_API_KEY"
```

Interactive chat:

```bash
constitutional-ai chat
constitutional-ai chat --execution-mode sequential
constitutional-ai chat --show-metrics
```

### 3. Run the GUI

Recommended:

```bash
python3 launch_gui.py
```

Alternative:

```bash
constitutional-ai-gui
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) if it does not auto-open.

The GUI keeps one top-level `Settings` tab and uses an internal left navigation for:

- credentials
- writer model
- judge model
- runtime parameters
- system prompts

When LiteLLM supports provider-side model discovery, the GUI uses a dropdown populated from `get_valid_models(check_provider_endpoint=True)`. Otherwise it falls back to manual text entry.

## Use in Python

```python
from constitutional_ai.config import load_config
from constitutional_ai.engine import run_constitutional_turn
from constitutional_ai.models import ChatMessage

config = load_config()
thread = [ChatMessage(role="user", content="Explain constitutional AI in 3 bullets.")]

turn = run_constitutional_turn(
    user_text=thread[-1].content,
    thread_messages=thread,
    config=config,
)

print(turn.final)
```

## Shared config

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

Set config values without editing JSON directly:

```bash
constitutional-ai config set --key settings.writer.provider --value anthropic
constitutional-ai config set --key settings.writer.model --value claude-sonnet-4-5-20250929
constitutional-ai config set --key settings.judge.provider --value openai
constitutional-ai config set --key settings.judge.model --value gpt-4o-mini
constitutional-ai config set --key settings.temperature --json-value 0.2
constitutional-ai config set --key settings.credentials.openai_api_key --value "sk-..."
```

Config shape:

```json
{
  "settings": {
    "credentials": {
      "openai_api_key": "",
      "anthropic_api_key": "",
      "gemini_api_key": "",
      "xai_api_key": "",
      "openrouter_api_key": "",
      "groq_api_key": "",
      "togetherai_api_key": "",
      "huggingface_api_key": "",
      "azure_api_key": ""
    },
    "writer": {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "api_base": "",
      "api_version": ""
    },
    "judge": {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "api_base": "",
      "api_version": ""
    },
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

Notes:

- Old OpenAI-only configs are migrated on load into the new structure.
- Environment variables override saved credentials for the matching provider.
- `api_base` is optional and works for custom endpoints or local servers.
- Writer and judge are configured independently.

## Examples

See [examples](./examples):

- `basic_single_turn/`
- `model_variants/`
- `history_management/`
- `constitution_variants/`
- `parallel_mode/`

## Notes

- Turn transcripts include run-stage events, token usage, duration, judge critiques, and required fixes.
- `max_iteration_ms` can stop long runs and return the latest revision.
- The engine no longer blocks execution on provider-side model listing; manual model entry works for unsupported providers and local models.
- Do not commit API keys. The tracked default config has been cleaned to remove embedded credentials.

## LLM-oriented docs

- `README.llm` is the structured repo companion for coding agents.
- Keep `README.md` and `README.llm` aligned after interface or config changes.
