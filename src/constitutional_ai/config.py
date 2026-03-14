"""Configuration loading and persistence for constitutional-ai."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse


DEFAULT_CONFIG_DIR = Path.home() / ".constitutional_ai"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.json"
DEFAULT_RULES = [
    "Be helpful, clear, and accurate.",
    "Do not provide illegal wrongdoing instructions or facilitation.",
    "Do not reveal secrets or private data (including API keys).",
    "If uncertain, state uncertainty and ask clarifying questions.",
    "Keep responses concise unless the user asks for depth.",
]


def normalize_base_url(value: str) -> str:
    """Normalize API base URL and reject malformed values."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Invalid base_url: value is empty.")

    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Invalid base_url: must start with http:// or https://.")
    if not parsed.netloc:
        raise ValueError("Invalid base_url: missing hostname.")

    path = parsed.path.rstrip("/")
    # Accept either root base URLs or values ending in /v1 and normalize both.
    while path.endswith("/v1"):
        path = path[: -len("/v1")]
        path = path.rstrip("/")
    if "/v1/" in path:
        raise ValueError("Invalid base_url: '/v1' may only appear at the end of the path.")

    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


@dataclass(slots=True)
class RuntimeSettings:
    """Runtime knobs used by the constitutional loop and model client."""

    api_key: str = ""
    base_url: str = "https://api.openai.com"
    writer_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    temperature: float = 0.4
    max_tokens: int = 650
    max_revisions_per_rule: int = 1
    execution_mode: str = "sequential"
    parallel_max_iterations: int = 0
    timeout_ms: int = 45_000

    @staticmethod
    def from_mapping(value: dict[str, Any] | None) -> "RuntimeSettings":
        """Create settings from untrusted mapping input with defaults."""
        value = value or {}
        execution_mode = str(value.get("execution_mode", "sequential") or "sequential").strip().lower()
        if execution_mode not in {"sequential", "parallel"}:
            execution_mode = "sequential"

        return RuntimeSettings(
            api_key=str(value.get("api_key", "") or ""),
            base_url=normalize_base_url(str(value.get("base_url", "https://api.openai.com") or "https://api.openai.com")),
            writer_model=str(value.get("writer_model", "gpt-4o-mini") or "gpt-4o-mini"),
            judge_model=str(value.get("judge_model", "gpt-4o-mini") or "gpt-4o-mini"),
            temperature=float(value.get("temperature", 0.4) or 0.4),
            max_tokens=int(value.get("max_tokens", 650) or 650),
            max_revisions_per_rule=int(value.get("max_revisions_per_rule", 1) or 1),
            execution_mode=execution_mode,
            parallel_max_iterations=max(0, int(value.get("parallel_max_iterations", 0) or 0)),
            timeout_ms=int(value.get("timeout_ms", 45_000) or 45_000),
        )


@dataclass(slots=True)
class PromptTemplates:
    """Prompt templates used by writer and judge roles."""

    writer_system: str = (
        "You are the writer agent. Write a helpful, safe, and accurate assistant response to the user's prompt. "
        "If you are revising, incorporate the judge's critique and follow the provided rule. "
        "Return ONLY the final user-facing answer, with no meta-commentary."
    )
    judge_pass_system: str = (
        "You are the judge agent. You evaluate a writer agent's answer against ONE rule at a time. "
        "Return JSON ONLY (no markdown, no extra text). First decide whether the rule applies to this user prompt "
        "and answer. If it does not apply, mark it as not applicable. If it applies, decide whether the answer "
        "follows the rule. Schema: {\"applies\": boolean, \"pass\": boolean}. "
        "Constraints: if applies is false, pass MUST be true."
    )
    judge_critique_system: str = (
        "You are the judge agent. You evaluate a writer agent's answer against ONE rule at a time. "
        "The answer already failed the rule. Provide critique and concrete required fixes. Return JSON ONLY "
        "(no markdown, no extra text). Schema: {\"critique\": string, \"required_fixes\": string}."
    )

    @staticmethod
    def from_mapping(value: dict[str, Any] | None) -> "PromptTemplates":
        """Create prompts from untrusted mapping input with defaults."""
        value = value or {}
        defaults = PromptTemplates()
        return PromptTemplates(
            writer_system=str(value.get("writer_system", defaults.writer_system) or defaults.writer_system),
            judge_pass_system=str(value.get("judge_pass_system", defaults.judge_pass_system) or defaults.judge_pass_system),
            judge_critique_system=str(
                value.get("judge_critique_system", defaults.judge_critique_system) or defaults.judge_critique_system
            ),
        )


@dataclass(slots=True)
class AppConfig:
    """Top-level persisted configuration shared by CLI, GUI, and notebooks."""

    settings: RuntimeSettings = field(default_factory=RuntimeSettings)
    rules: list[str] = field(default_factory=lambda: list(DEFAULT_RULES))
    prompts: PromptTemplates = field(default_factory=PromptTemplates)

    @staticmethod
    def from_mapping(value: dict[str, Any] | None) -> "AppConfig":
        """Create a complete config object from a JSON mapping."""
        value = value or {}
        rules_raw = value.get("rules", list(DEFAULT_RULES))
        rules = [str(line).strip() for line in rules_raw if str(line).strip()] if isinstance(rules_raw, list) else []
        return AppConfig(
            settings=RuntimeSettings.from_mapping(value.get("settings") if isinstance(value, dict) else None),
            rules=rules,
            prompts=PromptTemplates.from_mapping(value.get("prompts") if isinstance(value, dict) else None),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this config to a JSON-compatible dictionary."""
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    """Read JSON from disk, returning an empty mapping for missing or invalid files."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def get_api_key_source(path: str | Path | None = None) -> str:
    """Return where the effective API key comes from: env, config, or none."""
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return "environment"

    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    payload = _read_json(cfg_path)
    stored = ""
    if isinstance(payload.get("settings"), dict):
        stored = str(payload["settings"].get("api_key", "") or "").strip()
    return "config" if stored else "none"


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load config from disk and overlay environment-sourced API key."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    base = AppConfig.from_mapping(_read_json(cfg_path))
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        base.settings.api_key = env_key
    return base


def save_config(config: AppConfig, path: str | Path | None = None) -> Path:
    """Persist config to disk and return the final path used."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return cfg_path


def merge_config(base: AppConfig, payload: dict[str, Any] | None) -> AppConfig:
    """Merge untrusted override payload into an existing config object."""
    payload = payload or {}
    merged = AppConfig.from_mapping(base.to_dict())

    if isinstance(payload.get("settings"), dict):
        current = merged.settings
        incoming = payload["settings"]
        merged.settings = RuntimeSettings.from_mapping({**asdict(current), **incoming})

    if isinstance(payload.get("rules"), list):
        merged.rules = [str(line).strip() for line in payload["rules"] if str(line).strip()]

    if isinstance(payload.get("rules_text"), str):
        merged.rules = [line.strip() for line in payload["rules_text"].splitlines() if line.strip()]

    if isinstance(payload.get("prompts"), dict):
        merged.prompts = PromptTemplates.from_mapping({**asdict(merged.prompts), **payload["prompts"]})

    return merged
