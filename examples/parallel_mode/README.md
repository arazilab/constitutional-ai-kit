# Parallel Mode

This example shows how to run the constitutional loop in `parallel` mode.

## Run

From repository root:

```bash
python3 examples/parallel_mode/run.py
```

## What it demonstrates

- setting `settings.execution_mode = "parallel"`
- setting `settings.parallel_max_iterations = 1` (single rewrite round)
- setting `settings.parallel_max_iterations = 0` (continue until no rules are triggered)
- reading transcript data to inspect rewrite count and failed checks
