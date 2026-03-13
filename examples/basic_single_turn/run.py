"""Minimal single-turn constitutional AI example."""

from constitutional_ai.config import load_config
from constitutional_ai.engine import run_constitutional_turn
from constitutional_ai.models import ChatMessage


def main() -> None:
    """Run one prompt through the shared constitutional engine."""
    config = load_config()
    user_text = "Explain constitutional AI in three concise bullet points."
    thread = [ChatMessage(role="user", content=user_text)]

    turn = run_constitutional_turn(
        user_text=user_text,
        thread_messages=thread,
        config=config,
    )
    print(turn.final)


if __name__ == "__main__":
    main()
