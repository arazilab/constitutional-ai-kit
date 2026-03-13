"""Allow `python -m constitutional_ai` to invoke the CLI."""

from constitutional_ai.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
