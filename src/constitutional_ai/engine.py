"""Shared constitutional loop used by CLI, GUI API, and notebook workflows."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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


def _judge_pass_for_rule(
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout_ms: int,
    max_tokens: int,
    system_prompt: str,
    rule: str,
    thread_text: str,
    user_text: str,
    current: str,
) -> tuple[bool, bool, str, UsageStats]:
    """Run one rule pass-check and return normalized applies/pass flags plus raw payload."""
    pass_res = chat_completion(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
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
        max_tokens=max(256, min(800, max_tokens)),
        timeout_ms=timeout_ms,
    )

    pass_raw = pass_res.content.strip()
    pass_obj = _safe_json_parse(pass_raw)
    applies = bool(pass_obj["applies"]) if isinstance(pass_obj, dict) and "applies" in pass_obj else True
    passed = bool(pass_obj["pass"]) if isinstance(pass_obj, dict) and "pass" in pass_obj else False
    if not applies:
        passed = True
    return applies, passed, pass_raw, pass_res.usage


def _judge_critique_for_rule(
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout_ms: int,
    max_tokens: int,
    system_prompt: str,
    rule: str,
    thread_text: str,
    user_text: str,
    current: str,
) -> tuple[str, str, str, UsageStats]:
    """Run one rule critique and return parsed critique/fixes plus raw payload."""
    critique_res = chat_completion(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
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
        max_tokens=max(256, min(800, max_tokens)),
        timeout_ms=timeout_ms,
    )

    critique_raw = critique_res.content.strip()
    critique_obj = _safe_json_parse(critique_raw)
    critique = ""
    required_fixes = ""
    if isinstance(critique_obj, dict):
        critique = str(critique_obj.get("critique", "") or "")
        required_fixes = str(critique_obj.get("required_fixes", "") or "")

    if not critique:
        critique = "Judge output was not valid JSON. Please revise to satisfy the rule."
    if not required_fixes:
        required_fixes = "Revise the answer to satisfy the rule."
    return critique, required_fixes, critique_raw, critique_res.usage


def _writer_revision(
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout_ms: int,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    user_text: str,
    thread_text: str,
    current: str,
    critique: str,
    required_fixes: str,
) -> tuple[str, UsageStats]:
    """Ask writer to revise current answer from critique/fix guidance."""
    revision = chat_completion(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
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
                        "Judge critique:",
                        critique or "(No critique provided.)",
                        "",
                        "Required fixes:",
                        required_fixes or "(No required fixes provided.)",
                        "",
                        "Rewrite the answer to fully satisfy the rule(s) and the user's request.",
                    ]
                ),
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_ms=timeout_ms,
    )
    return revision.content.strip(), revision.usage


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

    if settings.execution_mode == "parallel":
        revision_rounds = 0
        while True:
            with ThreadPoolExecutor(max_workers=max(1, len(rules))) as executor:
                pass_results = list(
                    executor.map(
                        lambda pair: _judge_pass_for_rule(
                            api_key=settings.api_key,
                            base_url=settings.base_url,
                            model=settings.judge_model,
                            timeout_ms=settings.timeout_ms,
                            max_tokens=settings.max_tokens,
                            system_prompt=prompts.judge_pass_system,
                            rule=pair[1],
                            thread_text=thread_text,
                            user_text=user_text,
                            current=current,
                        ),
                        list(enumerate(rules)),
                    )
                )

            failed_rule_indices: list[int] = []
            checks_for_round: list[JudgeCheck] = []
            for rule_index, rule in enumerate(rules):
                applies, passed, pass_raw, pass_usage = pass_results[rule_index]
                check = JudgeCheck(
                    at=now_iso(),
                    rule_index=rule_index,
                    rule=rule,
                    applies=applies,
                    passed=passed,
                    pass_raw=pass_raw,
                    pass_usage=pass_usage,
                )
                checks_for_round.append(check)
                turn.usage.add(pass_usage)
                if applies and not passed:
                    failed_rule_indices.append(rule_index)

            turn.judge_checks.extend(checks_for_round)

            if not failed_rule_indices:
                break
            if settings.parallel_max_iterations > 0 and revision_rounds >= settings.parallel_max_iterations:
                break

            with ThreadPoolExecutor(max_workers=max(1, len(failed_rule_indices))) as executor:
                critique_results = list(
                    executor.map(
                        lambda idx: _judge_critique_for_rule(
                            api_key=settings.api_key,
                            base_url=settings.base_url,
                            model=settings.judge_model,
                            timeout_ms=settings.timeout_ms,
                            max_tokens=settings.max_tokens,
                            system_prompt=prompts.judge_critique_system,
                            rule=rules[idx],
                            thread_text=thread_text,
                            user_text=user_text,
                            current=current,
                        ),
                        failed_rule_indices,
                    )
                )

            for pos, rule_index in enumerate(failed_rule_indices):
                critique, required_fixes, critique_raw, critique_usage = critique_results[pos]
                check = checks_for_round[rule_index]
                check.critique = critique
                check.required_fixes = required_fixes
                check.critique_raw = critique_raw
                check.critique_usage = critique_usage
                turn.usage.add(critique_usage)

            combined_critique = "\n\n".join(
                [
                    "\n".join(
                        [
                            f"Rule {check.rule_index + 1}: {check.rule}",
                            f"Critique: {check.critique or '(No critique provided.)'}",
                            f"Required fixes: {check.required_fixes or '(No required fixes provided.)'}",
                        ]
                    )
                    for check in checks_for_round
                    if check.applies and not check.passed
                ]
            )
            combined_required_fixes = "\n".join(
                [
                    f"- Rule {check.rule_index + 1}: {check.required_fixes or 'Revise to satisfy this rule.'}"
                    for check in checks_for_round
                    if check.applies and not check.passed
                ]
            )

            current, revision_usage = _writer_revision(
                api_key=settings.api_key,
                base_url=settings.base_url,
                model=settings.writer_model,
                timeout_ms=settings.timeout_ms,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                system_prompt=prompts.writer_system,
                user_text=user_text,
                thread_text=thread_text,
                current=current,
                critique=combined_critique,
                required_fixes=combined_required_fixes,
            )
            turn.writer_drafts.append(
                WriterDraft(
                    at=now_iso(),
                    kind="revision",
                    content=current,
                    based_on_critique=combined_critique,
                    usage=revision_usage,
                )
            )
            turn.usage.add(revision_usage)
            revision_rounds += 1
    else:
        for rule_index, rule in enumerate(rules):
            revisions_for_rule = 0

            while True:
                applies, passed, pass_raw, pass_usage = _judge_pass_for_rule(
                    api_key=settings.api_key,
                    base_url=settings.base_url,
                    model=settings.judge_model,
                    timeout_ms=settings.timeout_ms,
                    max_tokens=settings.max_tokens,
                    system_prompt=prompts.judge_pass_system,
                    rule=rule,
                    thread_text=thread_text,
                    user_text=user_text,
                    current=current,
                )

                critique = ""
                required_fixes = ""
                critique_raw = ""
                critique_usage = UsageStats()

                if applies and not passed:
                    critique, required_fixes, critique_raw, critique_usage = _judge_critique_for_rule(
                        api_key=settings.api_key,
                        base_url=settings.base_url,
                        model=settings.judge_model,
                        timeout_ms=settings.timeout_ms,
                        max_tokens=settings.max_tokens,
                        system_prompt=prompts.judge_critique_system,
                        rule=rule,
                        thread_text=thread_text,
                        user_text=user_text,
                        current=current,
                    )
                    turn.usage.add(critique_usage)

                check = JudgeCheck(
                    at=now_iso(),
                    rule_index=rule_index,
                    rule=rule,
                    applies=applies,
                    passed=passed,
                    pass_raw=pass_raw,
                    pass_usage=pass_usage,
                    critique=critique,
                    required_fixes=required_fixes,
                    critique_raw=critique_raw,
                    critique_usage=critique_usage,
                )
                turn.judge_checks.append(check)
                turn.usage.add(pass_usage)

                if not applies or passed:
                    break
                if revisions_for_rule >= settings.max_revisions_per_rule:
                    break

                revisions_for_rule += 1
                current, revision_usage = _writer_revision(
                    api_key=settings.api_key,
                    base_url=settings.base_url,
                    model=settings.writer_model,
                    timeout_ms=settings.timeout_ms,
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens,
                    system_prompt=prompts.writer_system,
                    user_text=user_text,
                    thread_text=thread_text,
                    current=current,
                    critique=check.critique,
                    required_fixes=check.required_fixes,
                )
                turn.writer_drafts.append(
                    WriterDraft(
                        at=now_iso(),
                        kind="revision",
                        rule_index=rule_index,
                        rule=rule,
                        content=current,
                        based_on_critique=check.critique,
                        usage=revision_usage,
                    )
                )
                turn.usage.add(revision_usage)

    turn.final = current
    return turn
