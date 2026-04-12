# Model Variants

This example runs the same user request with multiple LiteLLM writer/judge model combinations.

## Run

From repository root:

```bash
python3 examples/model_variants/run.py
```

## What it demonstrates

- overriding provider + model settings per run
- keeping the same constitution and prompts
- comparing outputs by model pair

Update the `MODEL_PAIRS` list in `run.py` to match the providers and models available to your credentials.
