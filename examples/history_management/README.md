# History Management

This example keeps an explicit conversation history and reuses it across turns.

## Run

From repository root:

```bash
python3 examples/history_management/run.py
```

## What it demonstrates

- storing `user` and `assistant` messages in a single history list
- passing full history to each new turn
- producing context-aware responses over multiple turns
