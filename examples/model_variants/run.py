"""Compare outputs across different writer/judge model pairs."""

from constitutional_ai.config import load_config, merge_config
from constitutional_ai.engine import run_constitutional_turn
from constitutional_ai.models import ChatMessage


MODEL_PAIRS = [
    ("gpt-4o-mini", "gpt-4o-mini"),
    ("gpt-4o", "gpt-4o-mini"),
]


def main() -> None:
    """Run one prompt across configured model pairs and print each result."""
    base_config = load_config()
    user_text = "Draft a concise onboarding checklist for a new data scientist."

    for writer_model, judge_model in MODEL_PAIRS:
        config = merge_config(
            base_config,
            {
                "settings": {
                    "writer_model": writer_model,
                    "judge_model": judge_model,
                }
            },
        )

        thread = [ChatMessage(role="user", content=user_text)]
        turn = run_constitutional_turn(
            user_text=user_text,
            thread_messages=thread,
            config=config,
        )

        print(f"\n=== writer={writer_model} | judge={judge_model} ===")
        print(turn.final)


if __name__ == "__main__":
    main()
