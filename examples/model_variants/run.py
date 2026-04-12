"""Compare outputs across different LiteLLM writer/judge model pairs."""

from constitutional_ai.config import load_config, merge_config
from constitutional_ai.engine import run_constitutional_turn
from constitutional_ai.models import ChatMessage


MODEL_PAIRS = [
    (
        {"provider": "openai", "model": "gpt-4o-mini"},
        {"provider": "openai", "model": "gpt-4o-mini"},
    ),
    (
        {"provider": "openai", "model": "gpt-4o-mini"},
        {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
    ),
]


def main() -> None:
    """Run one prompt across configured model pairs and print each result."""
    base_config = load_config()
    user_text = "Draft a concise onboarding checklist for a new data scientist."

    for writer_settings, judge_settings in MODEL_PAIRS:
        config = merge_config(
            base_config,
            {
                "settings": {
                    "writer": writer_settings,
                    "judge": judge_settings,
                }
            },
        )

        thread = [ChatMessage(role="user", content=user_text)]
        turn = run_constitutional_turn(
            user_text=user_text,
            thread_messages=thread,
            config=config,
        )

        print(
            "\n=== "
            f"writer={writer_settings['provider']}/{writer_settings['model']} | "
            f"judge={judge_settings['provider']}/{judge_settings['model']} ==="
        )
        print(turn.final)


if __name__ == "__main__":
    main()
