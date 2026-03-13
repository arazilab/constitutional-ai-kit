"""Run one prompt under different constitutions for policy comparison."""

from constitutional_ai.config import load_config, merge_config
from constitutional_ai.engine import run_constitutional_turn
from constitutional_ai.models import ChatMessage


CONSTITUTIONS = {
    "concise_professional": [
        "Be accurate and practical.",
        "Use concise, high-signal wording.",
        "Do not include unnecessary background context.",
    ],
    "teaching_friendly": [
        "Be accurate and practical.",
        "Explain concepts step-by-step for beginners.",
        "Use short examples where helpful.",
    ],
    "safety_first": [
        "Be accurate and practical.",
        "Refuse harmful or illegal facilitation.",
        "When uncertain, explicitly state uncertainty.",
    ],
}


def main() -> None:
    """Execute one user prompt against multiple constitution rule sets."""
    base_config = load_config()
    user_text = "Give me a plan to evaluate a new LLM for customer support quality."

    for name, rules in CONSTITUTIONS.items():
        config = merge_config(base_config, {"rules": rules})
        thread = [ChatMessage(role="user", content=user_text)]
        turn = run_constitutional_turn(
            user_text=user_text,
            thread_messages=thread,
            config=config,
        )

        print(f"\n=== constitution={name} ===")
        print("Rules:")
        for rule in rules:
            print(f"- {rule}")
        print("\nAnswer:")
        print(turn.final)


if __name__ == "__main__":
    main()
