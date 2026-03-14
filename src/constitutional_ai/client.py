"""Minimal OpenAI Chat Completions client built on the Python standard library."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from constitutional_ai.config import normalize_base_url
from constitutional_ai.models import UsageStats


@dataclass(slots=True)
class CompletionResult:
    """Normalized result from a single model completion call."""

    content: str
    usage: UsageStats
    raw: dict[str, Any]


class OpenAIAPIError(RuntimeError):
    """Raised when the OpenAI-compatible API call fails."""


def _debug_enabled() -> bool:
    """Return True when debug logging is enabled by environment."""
    return os.getenv("CONSTITUTIONAL_AI_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _build_chat_completions_url(base_url: str) -> str:
    """Build the canonical chat-completions endpoint URL from normalized base URL."""
    normalized = normalize_base_url(base_url)
    return f"{normalized}/v1/chat/completions"


def chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_ms: int,
) -> CompletionResult:
    """Call `/v1/chat/completions` and return normalized content plus usage."""
    if not api_key:
        raise OpenAIAPIError("Missing API key. Set OPENAI_API_KEY or save it in config.")

    try:
        url = _build_chat_completions_url(base_url)
    except ValueError as exc:
        raise OpenAIAPIError(str(exc)) from exc

    if _debug_enabled():
        print(f"[constitutional_ai.debug] POST {url} model={model}", file=sys.stderr)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urlopen(request, timeout=max(1, timeout_ms // 1000)) as response:
            response_text = response.read().decode("utf-8")
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        try:
            parsed = json.loads(raw)
            message = parsed.get("error", {}).get("message") or raw
        except json.JSONDecodeError:
            message = raw or f"HTTP {exc.code}"
        if exc.code == 404:
            message = f"{message} (request URL: {url})"
        raise OpenAIAPIError(f"OpenAI API error ({exc.code}): {message}") from exc
    except URLError as exc:
        raise OpenAIAPIError(f"Network error while calling API: {exc}") from exc

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise OpenAIAPIError("API returned non-JSON response.") from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenAIAPIError("API response did not include choices[0].message.content.") from exc

    if not isinstance(content, str):
        raise OpenAIAPIError("API response content was not a string.")

    usage = UsageStats.from_mapping(data.get("usage"))
    return CompletionResult(content=content, usage=usage, raw=data)
