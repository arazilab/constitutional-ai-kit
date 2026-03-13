"""Utility helpers shared across entrypoints."""

from __future__ import annotations

from typing import Any

from constitutional_ai.models import ChatMessage


def sanitize_rules_text(text: str) -> list[str]:
    """Split newline-delimited rule text into a normalized list."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def normalize_chat_history(items: list[dict[str, Any]] | None) -> list[ChatMessage]:
    """Convert untrusted mapping data into normalized chat messages."""
    messages: list[ChatMessage] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        parsed = ChatMessage.from_mapping(item)
        if parsed is not None:
            messages.append(parsed)
    return messages
