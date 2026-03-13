# Examples

Each example is self-contained and has its own README.

## Prerequisites

From repository root:

```bash
pip install -e .
export OPENAI_API_KEY="sk-..."
```

## Scenarios

- `basic_single_turn/`: minimal one-turn usage
- `model_variants/`: run the same task with different writer/judge model pairs
- `history_management/`: maintain and reuse multi-turn chat history
- `constitution_variants/`: swap constitutions for different policy goals
