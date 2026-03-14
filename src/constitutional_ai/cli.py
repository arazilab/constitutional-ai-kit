"""Command-line entrypoint for constitutional-ai."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from constitutional_ai.config import AppConfig, DEFAULT_CONFIG_PATH, load_config, merge_config, save_config
from constitutional_ai.engine import run_constitutional_turn
from constitutional_ai.models import ChatMessage
from constitutional_ai.utils import sanitize_rules_text


def _load_rules_file(path: str | None) -> list[str] | None:
    """Load newline-delimited rules from a file path if provided."""
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8")
    return sanitize_rules_text(text)


def _run_once(args: argparse.Namespace) -> int:
    """Run a single constitutional turn and print the assistant answer."""
    config = load_config(args.config)
    rule_override = _load_rules_file(args.rules_file)

    overrides: dict[str, Any] = {}
    if rule_override is not None:
        overrides["rules"] = rule_override
    if args.api_key:
        overrides.setdefault("settings", {})["api_key"] = args.api_key
    if args.writer_model:
        overrides.setdefault("settings", {})["writer_model"] = args.writer_model
    if args.judge_model:
        overrides.setdefault("settings", {})["judge_model"] = args.judge_model
    if args.execution_mode:
        overrides.setdefault("settings", {})["execution_mode"] = args.execution_mode
    if args.parallel_max_iterations is not None:
        overrides.setdefault("settings", {})["parallel_max_iterations"] = args.parallel_max_iterations
    if args.max_iteration_ms is not None:
        overrides.setdefault("settings", {})["max_iteration_ms"] = args.max_iteration_ms

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
    overrides: dict[str, Any] = {}
    if args.execution_mode:
        overrides.setdefault("settings", {})["execution_mode"] = args.execution_mode
    if args.parallel_max_iterations is not None:
        overrides.setdefault("settings", {})["parallel_max_iterations"] = args.parallel_max_iterations
    if args.max_iteration_ms is not None:
        overrides.setdefault("settings", {})["max_iteration_ms"] = args.max_iteration_ms
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
    """Print the effective config JSON, optionally redacting API key."""
    config = load_config(args.config).to_dict()
    if args.redact_key and config.get("settings", {}).get("api_key"):
        config["settings"]["api_key"] = "***redacted***"
    print(json.dumps(config, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(description="Constitutional AI package CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one constitutional turn")
    run_parser.add_argument("--prompt", required=True, help="User prompt for this turn")
    run_parser.add_argument("--config", default=None, help="Path to config JSON")
    run_parser.add_argument("--rules-file", default=None, help="Optional newline-delimited rules file")
    run_parser.add_argument("--api-key", default=None, help="Optional API key override")
    run_parser.add_argument("--writer-model", default=None, help="Optional writer model override")
    run_parser.add_argument("--judge-model", default=None, help="Optional judge model override")
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
    config_show.add_argument("--redact-key", action="store_true", help="Mask API key in output")
    config_show.set_defaults(func=_config_show)

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
