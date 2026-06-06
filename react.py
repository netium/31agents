import os

from dotenv import load_dotenv
from openai import OpenAI

from agents import ReactAgent

load_dotenv()


def add(a: int, b: int) -> int:
    """Add two numbers together and return the result.

    Args:
        a: The first integer to add.
        b: The second integer to add.
    """
    return a + b


def get_weather(location: str) -> str:
    """Get the current weather for a given location.

    Args:
        location: The location to get the weather for.
    """
    return f"The current weather in {location} is sunny with a temperature of 25°C."


def main():
    llm = OpenAI(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("OLLAMA_API_KEY", "ollama")
    )

    agent = ReactAgent(llm, tools=[add, get_weather])

    history: list[dict[str, str]] = []

    print("ReAct Agent — type '/exit', '/quit', or '/bye' to stop, or press Ctrl-D / Ctrl-C to end.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit", "/bye"):
            print("Goodbye!")
            break

        answer = agent.call(user_input, history=history)

        print(f"\nAssistant: {answer}")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
