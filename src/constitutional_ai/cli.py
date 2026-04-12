"""Command-line entrypoint for constitutional-ai."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from constitutional_ai.config import AppConfig, DEFAULT_CONFIG_PATH, load_config, merge_config, save_config, set_config_value
from constitutional_ai.engine import run_constitutional_turn
from constitutional_ai.models import ChatMessage
from constitutional_ai.utils import sanitize_rules_text


def _load_rules_file(path: str | None) -> list[str] | None:
    """Load newline-delimited rules from a file path if provided."""
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8")
    return sanitize_rules_text(text)


def _redact_credentials(payload: dict[str, Any]) -> dict[str, Any]:
    """Mask saved credentials in a config payload."""
    settings = payload.get("settings")
    if not isinstance(settings, dict):
        return payload
    credentials = settings.get("credentials")
    if isinstance(credentials, dict):
        for key, value in list(credentials.items()):
            if str(value or "").strip():
                credentials[key] = "***redacted***"
    return payload


def _apply_common_runtime_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Build provider/model/credential overrides from CLI flags."""
    overrides: dict[str, Any] = {"settings": {"credentials": {}, "writer": {}, "judge": {}}}
    settings = overrides["settings"]

    for field_name in [
        "openai_api_key",
        "anthropic_api_key",
        "gemini_api_key",
        "xai_api_key",
        "openrouter_api_key",
        "groq_api_key",
        "togetherai_api_key",
        "huggingface_api_key",
        "azure_api_key",
    ]:
        value = getattr(args, field_name, None)
        if value:
            settings["credentials"][field_name] = value

    for role in ["writer", "judge"]:
        for field_name in ["provider", "model", "api_base", "api_version"]:
            value = getattr(args, f"{role}_{field_name}", None)
            if value is not None:
                settings[role][field_name] = value

    if args.execution_mode:
        settings["execution_mode"] = args.execution_mode
    if args.parallel_max_iterations is not None:
        settings["parallel_max_iterations"] = args.parallel_max_iterations
    if args.max_iteration_ms is not None:
        settings["max_iteration_ms"] = args.max_iteration_ms
    if args.temperature is not None:
        settings["temperature"] = args.temperature
    if args.max_tokens is not None:
        settings["max_tokens"] = args.max_tokens
    if args.timeout_ms is not None:
        settings["timeout_ms"] = args.timeout_ms

    if not settings["credentials"]:
        settings.pop("credentials")
    if not settings["writer"]:
        settings.pop("writer")
    if not settings["judge"]:
        settings.pop("judge")
    if not settings:
        return {}
    return overrides


def _run_once(args: argparse.Namespace) -> int:
    """Run a single constitutional turn and print the assistant answer."""
    config = load_config(args.config)
    overrides = _apply_common_runtime_overrides(args)
    rule_override = _load_rules_file(args.rules_file)
    if rule_override is not None:
        overrides["rules"] = rule_override

    merged = merge_config(config, overrides)
    user_msg = ChatMessage(role="user", content=args.prompt)
    turn = run_constitutional_turn(user_text=args.prompt, thread_messages=[user_msg], config=merged)

    if args.json:
        print(json.dumps(turn.to_dict(), indent=2))
    else:
        print(turn.final)
        if args.show_metrics:
            print(f"\n[metrics] duration_ms={turn.duration_ms} total_tokens={turn.usage.total_tokens}")
    return 0


def _chat_loop(args: argparse.Namespace) -> int:
    """Run a local REPL chat loop using the shared constitutional engine."""
    config = load_config(args.config)
    overrides = _apply_common_runtime_overrides(args)
    if overrides:
        config = merge_config(config, overrides)
    history: list[ChatMessage] = []

    print("Constitutional AI chat. Type '/exit' to quit.")
    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "exit", "quit"}:
            break

        history.append(ChatMessage(role="user", content=user_text))
        turn = run_constitutional_turn(user_text=user_text, thread_messages=history, config=config)
        print(f"assistant> {turn.final}")
        if args.show_metrics:
            print(f"[metrics] duration_ms={turn.duration_ms} total_tokens={turn.usage.total_tokens}")
        history.append(ChatMessage(role="assistant", content=turn.final))

    return 0


def _config_init(args: argparse.Namespace) -> int:
    """Write a starter config file if one does not already exist."""
    path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    if path.exists() and not args.force:
        print(f"Config already exists: {path}")
        print("Use --force to overwrite.")
        return 1

    config = AppConfig()
    save_config(config, path)
    print(f"Wrote config: {path}")
    return 0


def _config_show(args: argparse.Namespace) -> int:
    """Print the effective config JSON, optionally redacting credentials."""
    config = load_config(args.config).to_dict()
    if args.redact_key:
        config = _redact_credentials(config)
    print(json.dumps(config, indent=2))
    return 0


def _config_set(args: argparse.Namespace) -> int:
    """Set one config key by dotted path and persist the updated config."""
    if args.value is None and args.json_value is None:
        raise ValueError("Provide either --value or --json-value.")
    if args.value is not None and args.json_value is not None:
        raise ValueError("Use only one of --value or --json-value.")

    config = load_config(args.config)
    value = json.loads(args.json_value) if args.json_value is not None else args.value
    updated = set_config_value(config, args.key, value)
    path = save_config(updated, args.config)
    print(f"Updated {args.key} in {path}")
    if args.show:
        payload = updated.to_dict()
        if args.redact_key:
            payload = _redact_credentials(payload)
        print(json.dumps(payload, indent=2))
    return 0


def _add_provider_flags(parser: argparse.ArgumentParser) -> None:
    """Add provider credential and endpoint override flags to a parser."""
    for field_name, help_text in [
        ("openai_api_key", "Override the OpenAI API key"),
        ("anthropic_api_key", "Override the Anthropic API key"),
        ("gemini_api_key", "Override the Gemini API key"),
        ("xai_api_key", "Override the xAI API key"),
        ("openrouter_api_key", "Override the OpenRouter API key"),
        ("groq_api_key", "Override the Groq API key"),
        ("togetherai_api_key", "Override the Together AI API key"),
        ("huggingface_api_key", "Override the Hugging Face API key"),
        ("azure_api_key", "Override the Azure OpenAI API key"),
    ]:
        parser.add_argument(f"--{field_name.replace('_', '-')}", dest=field_name, default=None, help=help_text)

    for role in ["writer", "judge"]:
        parser.add_argument(
            f"--{role}-provider",
            dest=f"{role}_provider",
            default=None,
            help=f"LiteLLM provider for the {role} role",
        )
        parser.add_argument(
            f"--{role}-model",
            dest=f"{role}_model",
            default=None,
            help=f"Model name for the {role} role",
        )
        parser.add_argument(
            f"--{role}-api-base",
            dest=f"{role}_api_base",
            default=None,
            help=f"Optional API base for the {role} role",
        )
        parser.add_argument(
            f"--{role}-api-version",
            dest=f"{role}_api_version",
            default=None,
            help=f"Optional API version for the {role} role",
        )

    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max tokens")
    parser.add_argument("--timeout-ms", type=int, default=None, help="Override request timeout in milliseconds")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(description="Constitutional AI package CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one constitutional turn")
    run_parser.add_argument("--prompt", required=True, help="User prompt for this turn")
    run_parser.add_argument("--config", default=None, help="Path to config JSON")
    run_parser.add_argument("--rules-file", default=None, help="Optional newline-delimited rules file")
    _add_provider_flags(run_parser)
    run_parser.add_argument(
        "--execution-mode",
        choices=["sequential", "parallel"],
        default=None,
        help="Override constitutional execution mode",
    )
    run_parser.add_argument(
        "--parallel-max-iterations",
        type=int,
        default=None,
        help="Parallel mode rewrite cap (0 means run until no rules fail)",
    )
    run_parser.add_argument(
        "--max-iteration-ms",
        type=int,
        default=None,
        help="Stop constitutional loop after this many milliseconds (0 means no limit)",
    )
    run_parser.add_argument("--show-metrics", action="store_true", help="Print duration/token metrics")
    run_parser.add_argument("--json", action="store_true", help="Print full transcript JSON")
    run_parser.set_defaults(func=_run_once)

    chat_parser = subparsers.add_parser("chat", help="Interactive REPL chat")
    chat_parser.add_argument("--config", default=None, help="Path to config JSON")
    _add_provider_flags(chat_parser)
    chat_parser.add_argument(
        "--execution-mode",
        choices=["sequential", "parallel"],
        default=None,
        help="Override constitutional execution mode for this chat session",
    )
    chat_parser.add_argument(
        "--parallel-max-iterations",
        type=int,
        default=None,
        help="Parallel mode rewrite cap (0 means run until no rules fail)",
    )
    chat_parser.add_argument(
        "--max-iteration-ms",
        type=int,
        default=None,
        help="Stop constitutional loop after this many milliseconds (0 means no limit)",
    )
    chat_parser.add_argument("--show-metrics", action="store_true", help="Print duration/token metrics per turn")
    chat_parser.set_defaults(func=_chat_loop)

    config_parser = subparsers.add_parser("config", help="Manage config")
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)

    config_init = config_sub.add_parser("init", help="Create starter config")
    config_init.add_argument("--config", default=None, help="Path to config JSON")
    config_init.add_argument("--force", action="store_true", help="Overwrite existing config")
    config_init.set_defaults(func=_config_init)

    config_show = config_sub.add_parser("show", help="Show effective config")
    config_show.add_argument("--config", default=None, help="Path to config JSON")
    config_show.add_argument("--redact-key", action="store_true", help="Mask provider API keys in output")
    config_show.set_defaults(func=_config_show)

    config_set = config_sub.add_parser("set", help="Set one config value by dotted path")
    config_set.add_argument("--config", default=None, help="Path to config JSON")
    config_set.add_argument("--key", required=True, help="Dotted key path (example: settings.writer.model)")
    config_set.add_argument(
        "--value",
        default=None,
        help="Raw string value to set (for typed values use --json-value)",
    )
    config_set.add_argument(
        "--json-value",
        default=None,
        help="JSON value to set (example: 0.2, 650, true, [\"Rule A\",\"Rule B\"])",
    )
    config_set.add_argument("--show", action="store_true", help="Print config after update")
    config_set.add_argument("--redact-key", action="store_true", help="Mask provider API keys when used with --show")
    config_set.set_defaults(func=_config_set)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI program entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
