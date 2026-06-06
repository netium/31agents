import json
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def add(a: int, b: int) -> int:
    """Add two numbers together and return the result."""
    return a + b

def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    # This is a mock implementation. In a real implementation, you would call a weather API.
    return f"The current weather in {location} is sunny with a temperature of 25°C."

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers together and return the result as an integer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "The first integer to add."
                    },
                    "b": {
                        "type": "integer",
                        "description": "The second integer to add."
                    }
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for."
                    }
                },
                "required": ["location"]
            }
        }
    }
]

available_tools = {
    "add": add,
    "get_weather": get_weather
}

class ReactAgent:
    def __init__(self, llm: OpenAI):
        self.llm = llm

    def call(self, input: str, history: list[dict[str, str]] = None) -> str:

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant, you can call functions to get information or perform tasks, you shall always prefer to use the tools provided to you if needed. If you don't know the answer, say you don't know."
            }
        ]

        if history:
            messages.extend(history)

        messages.append({
            "role": "user",
            "content": input
        })

        iter_num = 1

        print("--Start react loop--")

        while True:
            print(f'ReAct Loop Iteration #: {iter_num}:')
            response = self.llm.chat.completions.create(
                model=os.getenv("OLLAMA_LLM_MODEL", "qwen3.6:35b"),
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
                temperature=1
            )

            reponse_message = response.choices[0].message

            print(f"LLM response: {reponse_message.content}")

            if reponse_message.tool_calls:
                tool_call_response = self.handle_tool_calls(reponse_message.tool_calls, available_tools)
                messages.append(reponse_message)
                messages.append({
                    "role": "tool",
                    "tool_call_id" : tool_call_response['tool_call_id'],
                    "name": tool_call_response['name'],
                    "content": str(tool_call_response['content'])
                })
            else:
                print("--End react loop--")
                print(f"Final message list: {str(messages)}")

                return reponse_message.content

            iter_num += 1

    def handle_tool_calls(self, tool_calls, available_tools) -> dict:
        for tool_call in tool_calls:
            function_name = tool_call.function.name

            function_args = json.loads(tool_call.function.arguments)

            print(f"Calling tool request: {function_name} with args: {function_args}")

            if function_name in available_tools:
                function_to_call = available_tools[function_name]
                function_response = function_to_call(**function_args)
                print(f"Calling tool response: {function_response}")
                return {
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                }
            else:
                print(f"Calling tool error: Tool {function_name} not found.")
                return {
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": f"Tool {function_name} not found."
                }

def main():
    llm = OpenAI(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama"
    )

    agent = ReactAgent(llm)

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