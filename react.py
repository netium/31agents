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


def headlight_control(action: str, mode: str = "low", intensity: int = 100) -> str:
    """Control the vehicle's headlights.

    Args:
        action: One of 'on' or 'off'.
        mode: Beam mode — 'low', 'high', or 'auto'. Ignored when action is 'off'.
        intensity: Brightness percentage from 0 to 100. Ignored when action is 'off'.
    """
    if action == "off":
        return "Headlights turned off."
    return f"Headlights turned on ({mode} beam, {intensity}% intensity)."


def window_control(position: int, window: str = "driver") -> str:
    """Control the vehicle's windows.

    Args:
        position: Target position percentage from 0 (fully closed) to 100 (fully open).
        window: Which window to control — 'driver', 'passenger', 'rear_left',
            'rear_right', or 'all'.
    """
    return f"{window} window set to {position}% open."


def airconditioner_control(action: str, temperature: int = 22, mode: str = "auto") -> str:
    """Control the vehicle's air conditioner.

    Args:
        action: One of 'on' or 'off'.
        temperature: Target cabin temperature in degrees Celsius. Ignored when
            action is 'off'.
        mode: One of 'cool', 'heat', 'auto', or 'fan'. Ignored when action is 'off'.
    """
    if action == "off":
        return "Air conditioner turned off."
    return f"Air conditioner turned on — target {temperature}°C, mode: {mode}."


def seat_heating_control(seat: str, level: str) -> str:
    """Control the heating of a vehicle seat.

    Args:
        seat: Which seat to heat — 'driver' or 'passenger'.
        level: Heating level — 'off', 'low', 'medium', or 'high'.
    """
    return f"{seat.capitalize()} seat heating set to {level}."


def seat_massage_control(seat: str, action: str, mode: str = "wave") -> str:
    """Control the massage function of a vehicle seat.

    Args:
        seat: Which seat — 'driver' or 'passenger'.
        action: One of 'on' or 'off'.
        mode: Massage mode — 'wave', 'pulse', or 'roll'. Ignored when action is 'off'.
    """
    if action == "off":
        return f"{seat.capitalize()} seat massage turned off."
    return f"{seat.capitalize()} seat massage turned on ({mode} mode)."


def sunroof_control(position: int) -> str:
    """Control the vehicle's sunroof.

    Args:
        position: Target position percentage from 0 (fully closed) to 100 (fully open).
    """
    return f"Sunroof set to {position}% open."


def volume_control(action: str, level: int = 0) -> str:
    """Control the vehicle's media volume.

    Args:
        action: One of 'set', 'increase', 'decrease', or 'mute'.
        level: Target volume percentage from 0 to 100 when action is 'set';
            otherwise the delta to apply for 'increase' or 'decrease'.
    """
    if action == "mute":
        return "Audio muted."
    if action == "set":
        return f"Volume set to {level}%."
    if action == "increase":
        return f"Volume increased by {level}%."
    if action == "decrease":
        return f"Volume decreased by {level}%."
    return f"Unknown volume action: {action}."


def main():
    llm = OpenAI(
        base_url=os.getenv("OPENAPI_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("OPENAPI_API_KEY", "ollama")
    )

    agent = ReactAgent(
        llm,
        tools=[
            add,
            get_weather,
            headlight_control,
            window_control,
            airconditioner_control,
            seat_heating_control,
            seat_massage_control,
            sunroof_control,
            volume_control,
        ],
    )

    history: list[dict[str, str]] = []

    print(
        "ReAct Vehicle Assistant — type '/exit', '/quit', or '/bye' to stop, "
        "'/messages' to show the conversation history, "
        "or press Ctrl-D / Ctrl-C to end."
    )

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

        if user_input.lower() == "/messages":
            if not history:
                print("\nAssistant: (no messages yet)")
            else:
                print("\n--- Conversation history ---")
                for i, msg in enumerate(history, start=1):
                    print(f"[{i}] {msg['role']}: {msg['content']}")
                print("--- end of history ---")
            continue

        answer = agent.call(user_input, history=history)

        print(f"\nAssistant: {answer}")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
