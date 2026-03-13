"""Multi-turn history example using a shared conversation thread."""

from constitutional_ai.config import load_config
from constitutional_ai.engine import run_constitutional_turn
from constitutional_ai.models import ChatMessage


USER_TURNS = [
    "I am preparing a machine learning course. Suggest 5 modules.",
    "Now add one practical assignment per module.",
    "Make the whole plan fit a 4-week schedule.",
]


def main() -> None:
    """Run multiple turns while preserving full chat history."""
    config = load_config()
    history: list[ChatMessage] = []

    for user_text in USER_TURNS:
        history.append(ChatMessage(role="user", content=user_text))
        turn = run_constitutional_turn(
            user_text=user_text,
            thread_messages=history,
            config=config,
        )

        assistant_text = turn.final
        history.append(ChatMessage(role="assistant", content=assistant_text))

        print("\nUSER:")
        print(user_text)
        print("\nASSISTANT:")
        print(assistant_text)


if __name__ == "__main__":
    main()
