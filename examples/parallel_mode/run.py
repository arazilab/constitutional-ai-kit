"""Demonstrate parallel constitutional mode with bounded and unbounded iteration settings."""

from constitutional_ai.config import load_config, merge_config
from constitutional_ai.engine import run_constitutional_turn
from constitutional_ai.models import ChatMessage


RULES = [
    "Be accurate and practical.",
    "Keep the answer concise (4 short bullet points max).",
    "Include one concrete example.",
]


def _count_failed_checks(turn_dict: dict) -> int:
    """Return count of applicable failed checks in a transcript dictionary."""
    checks = turn_dict.get("judge", {}).get("checks", [])
    return sum(1 for check in checks if check.get("applies", True) and not check.get("pass", False))


def _run_case(*, label: str, parallel_max_iterations: int) -> None:
    """Run one parallel-mode case and print a compact summary."""
    base_config = load_config()
    config = merge_config(
        base_config,
        {
            "rules": RULES,
            "settings": {
                "execution_mode": "parallel",
                "parallel_max_iterations": parallel_max_iterations,
            },
        },
    )

    user_text = "Write guidance for evaluating a new customer support chatbot."
    thread = [ChatMessage(role="user", content=user_text)]
    turn = run_constitutional_turn(user_text=user_text, thread_messages=thread, config=config)
    turn_dict = turn.to_dict()

    failed_checks = _count_failed_checks(turn_dict)
    revision_count = sum(1 for draft in turn_dict.get("writer", {}).get("drafts", []) if draft.get("kind") == "revision")

    print(f"\n=== {label} ===")
    print(f"parallel_max_iterations: {parallel_max_iterations}")
    print(f"writer revisions: {revision_count}")
    print(f"failed checks in transcript: {failed_checks}")
    print("\nFinal answer:\n")
    print(turn.final)


def main() -> None:
    """Run bounded and unbounded parallel-mode examples."""
    _run_case(label="Parallel (single rewrite round)", parallel_max_iterations=1)
    _run_case(label="Parallel (until no rule fails)", parallel_max_iterations=0)


if __name__ == "__main__":
    main()
