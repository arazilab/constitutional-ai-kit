"""Shared constitutional loop used by CLI, GUI API, and notebook workflows."""

from __future__ import annotations

import json
from typing import Any

from constitutional_ai.client import chat_completion
from constitutional_ai.config import AppConfig
from constitutional_ai.models import ChatMessage, JudgeCheck, TurnTranscript, UsageStats, WriterDraft, now_iso


def _safe_json_parse(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model output; return None if parsing fails."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _format_thread_for_prompt(messages: list[ChatMessage]) -> str:
    """Format thread history into a stable text block for judge and revision prompts."""
    if not messages:
        return "(empty)"
    chunks: list[str] = []
    for index, msg in enumerate(messages, start=1):
        chunks.append(f"[{index}] {msg.role}\n{msg.content}")
    return "\n\n".join(chunks)


def _collect_message_list(thread_messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Convert internal messages to OpenAI messages format."""
    return [msg.to_openai() for msg in thread_messages]


def run_constitutional_turn(
    *,
    user_text: str,
    thread_messages: list[ChatMessage],
    config: AppConfig,
) -> TurnTranscript:
    """Run one writer/judge turn and return a full transcript object."""
    settings = config.settings
    prompts = config.prompts
    rules = [line.strip() for line in config.rules if line.strip()]

    thread = _collect_message_list(thread_messages)
    turn = TurnTranscript(user=user_text, thread=thread, rules=rules)

    initial = chat_completion(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.writer_model,
        messages=[{"role": "system", "content": prompts.writer_system}, *thread],
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout_ms=settings.timeout_ms,
    )
    current = initial.content.strip()
    turn.writer_drafts.append(
        WriterDraft(
            at=now_iso(),
            kind="initial",
            content=current,
            usage=initial.usage,
        )
    )
    turn.usage.add(initial.usage)

    thread_text = _format_thread_for_prompt(thread_messages)

    for rule_index, rule in enumerate(rules):
        revisions_for_rule = 0

        while True:
            pass_res = chat_completion(
                api_key=settings.api_key,
                base_url=settings.base_url,
                model=settings.judge_model,
                messages=[
                    {"role": "system", "content": prompts.judge_pass_system},
                    {
                        "role": "user",
                        "content": "\n".join(
                            [
                                f"Rule: {rule}",
                                "",
                                "Conversation thread (oldest -> newest):",
                                thread_text,
                                "",
                                "Latest user message:",
                                user_text,
                                "",
                                "User prompt:",
                                user_text,
                                "",
                                "Writer answer:",
                                current,
                            ]
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=max(256, min(800, settings.max_tokens)),
                timeout_ms=settings.timeout_ms,
            )

            pass_obj = _safe_json_parse(pass_res.content.strip())
            applies = bool(pass_obj["applies"]) if isinstance(pass_obj, dict) and "applies" in pass_obj else True
            passed = bool(pass_obj["pass"]) if isinstance(pass_obj, dict) and "pass" in pass_obj else False
            if not applies:
                passed = True

            critique = ""
            required_fixes = ""
            critique_raw = ""
            critique_usage = UsageStats()

            if applies and not passed:
                critique_res = chat_completion(
                    api_key=settings.api_key,
                    base_url=settings.base_url,
                    model=settings.judge_model,
                    messages=[
                        {"role": "system", "content": prompts.judge_critique_system},
                        {
                            "role": "user",
                            "content": "\n".join(
                                [
                                    f"Rule: {rule}",
                                    "",
                                    "Conversation thread (oldest -> newest):",
                                    thread_text,
                                    "",
                                    "Latest user message:",
                                    user_text,
                                    "",
                                    "User prompt:",
                                    user_text,
                                    "",
                                    "Writer answer:",
                                    current,
                                    "",
                                    "Explain what is wrong with the answer relative to the rule and what must be changed.",
                                ]
                            ),
                        },
                    ],
                    temperature=0.0,
                    max_tokens=max(256, min(800, settings.max_tokens)),
                    timeout_ms=settings.timeout_ms,
                )
                critique_raw = critique_res.content.strip()
                critique_usage = critique_res.usage
                turn.usage.add(critique_usage)

                critique_obj = _safe_json_parse(critique_raw)
                if isinstance(critique_obj, dict):
                    critique = str(critique_obj.get("critique", "") or "")
                    required_fixes = str(critique_obj.get("required_fixes", "") or "")

                if not critique:
                    critique = "Judge output was not valid JSON. Please revise to satisfy the rule."
                if not required_fixes:
                    required_fixes = "Revise the answer to satisfy the rule."

            check = JudgeCheck(
                at=now_iso(),
                rule_index=rule_index,
                rule=rule,
                applies=applies,
                passed=passed,
                pass_raw=pass_res.content.strip(),
                pass_usage=pass_res.usage,
                critique=critique,
                required_fixes=required_fixes,
                critique_raw=critique_raw,
                critique_usage=critique_usage,
            )
            turn.judge_checks.append(check)
            turn.usage.add(pass_res.usage)

            if not applies or passed:
                break
            if revisions_for_rule >= settings.max_revisions_per_rule:
                break

            revisions_for_rule += 1
            revision = chat_completion(
                api_key=settings.api_key,
                base_url=settings.base_url,
                model=settings.writer_model,
                messages=[
                    {"role": "system", "content": prompts.writer_system},
                    {
                        "role": "user",
                        "content": "\n".join(
                            [
                                "User prompt:",
                                user_text,
                                "",
                                "Conversation thread (oldest -> newest):",
                                thread_text,
                                "",
                                "Current draft answer:",
                                current,
                                "",
                                "Judge critique (for one specific rule):",
                                check.critique or "(No critique provided.)",
                                "",
                                "Required fixes:",
                                check.required_fixes or "(No required fixes provided.)",
                                "",
                                "Rewrite the answer to fully satisfy the rule and the user's request.",
                            ]
                        ),
                    },
                ],
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                timeout_ms=settings.timeout_ms,
            )
            current = revision.content.strip()
            turn.writer_drafts.append(
                WriterDraft(
                    at=now_iso(),
                    kind="revision",
                    rule_index=rule_index,
                    rule=rule,
                    content=current,
                    based_on_critique=check.critique,
                    usage=revision.usage,
                )
            )
            turn.usage.add(revision.usage)

    turn.final = current
    return turn
