"""Minimal OpenAI Chat Completions client built on the Python standard library."""

from __future__ import annotations

import json
import os
import ssl
import sys
import time
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


_MODEL_CACHE: dict[tuple[str, str], tuple[float, list[dict[str, Any]]]] = {}
_MODEL_CACHE_TTL_SECONDS = 60.0


def _debug_enabled() -> bool:
    """Return True when debug logging is enabled by environment."""
    return os.getenv("CONSTITUTIONAL_AI_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _build_chat_completions_url(base_url: str) -> str:
    """Build the canonical chat-completions endpoint URL from normalized base URL."""
    normalized = normalize_base_url(base_url)
    return f"{normalized}/v1/chat/completions"


def _build_models_url(base_url: str) -> str:
    """Build the canonical models endpoint URL from normalized base URL."""
    normalized = normalize_base_url(base_url)
    return f"{normalized}/v1/models"


def _build_ssl_context() -> ssl.SSLContext:
    """Build TLS context, preferring explicit CA bundle settings and certifi when available."""
    bundle_path = os.getenv("SSL_CERT_FILE", "").strip() or os.getenv("REQUESTS_CA_BUNDLE", "").strip()
    if bundle_path:
        return ssl.create_default_context(cafile=bundle_path)

    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


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
        ssl_context = _build_ssl_context()
        with urlopen(request, timeout=max(1, timeout_ms // 1000), context=ssl_context) as response:
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
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(exc):
            raise OpenAIAPIError(
                "Network error while calling API: TLS certificate verification failed. "
                "Try installing/updating CA certificates (for example: `pip install certifi`) and/or set "
                "`SSL_CERT_FILE` to a valid CA bundle path."
            ) from exc
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


def list_models(*, api_key: str, base_url: str, timeout_ms: int) -> list[dict[str, Any]]:
    """Return available model metadata from `/v1/models`."""
    if not api_key:
        raise OpenAIAPIError("Missing API key. Set OPENAI_API_KEY or save it in config.")

    try:
        url = _build_models_url(base_url)
    except ValueError as exc:
        raise OpenAIAPIError(str(exc)) from exc

    cache_key = (base_url, api_key)
    now = time.time()
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None and now - cached[0] <= _MODEL_CACHE_TTL_SECONDS:
        return list(cached[1])

    if _debug_enabled():
        print(f"[constitutional_ai.debug] GET {url}", file=sys.stderr)

    request = Request(
        url=url,
        method="GET",
        headers={
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        ssl_context = _build_ssl_context()
        with urlopen(request, timeout=max(1, timeout_ms // 1000), context=ssl_context) as response:
            response_text = response.read().decode("utf-8")
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        try:
            parsed = json.loads(raw)
            message = parsed.get("error", {}).get("message") or raw
        except json.JSONDecodeError:
            message = raw or f"HTTP {exc.code}"
        raise OpenAIAPIError(f"OpenAI API error ({exc.code}): {message}") from exc
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(exc):
            raise OpenAIAPIError(
                "Network error while calling API: TLS certificate verification failed. "
                "Try installing/updating CA certificates (for example: `pip install certifi`) and/or set "
                "`SSL_CERT_FILE` to a valid CA bundle path."
            ) from exc
        raise OpenAIAPIError(f"Network error while calling API: {exc}") from exc

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise OpenAIAPIError("API returned non-JSON response for model listing.") from exc

    raw_models = payload.get("data", [])
    if not isinstance(raw_models, list):
        raise OpenAIAPIError("API response for model listing was malformed.")

    models: list[dict[str, Any]] = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id", "") or "").strip()
        if not model_id:
            continue
        models.append(
            {
                "id": model_id,
                "created": int(item.get("created", 0) or 0),
                "object": str(item.get("object", "model") or "model"),
                "owned_by": str(item.get("owned_by", "") or ""),
            }
        )

    models.sort(key=lambda m: m["id"])
    _MODEL_CACHE[cache_key] = (now, list(models))
    return models
