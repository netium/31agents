import inspect
import json
import os
from collections.abc import Callable
from typing import get_type_hints

from openai import OpenAI

from agents._tools import parse_param_descriptions, python_type_to_json_schema


class ReactAgent:
    def __init__(self, llm: OpenAI, tools: list[Callable]):
        self.llm = llm
        self._available_tools: dict[str, Callable] = {func.__name__: func for func in tools}
        self._tools_schema: list[dict] = self._build_tools_schema(tools)

    def _build_tools_schema(self, tools: list[Callable]) -> list[dict]:
        schema: list[dict] = []
        for func in tools:
            sig = inspect.signature(func)
            try:
                hints = get_type_hints(func)
            except Exception:
                hints = dict(getattr(func, "__annotations__", {}))

            param_descriptions = parse_param_descriptions(func.__doc__)

            properties: dict[str, dict] = {}
            required: list[str] = []
            for name, param in sig.parameters.items():
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                tp = hints.get(name, str)
                properties[name] = {
                    "type": python_type_to_json_schema(tp),
                    "description": param_descriptions.get(name, ""),
                }
                if param.default is inspect.Parameter.empty:
                    required.append(name)

            description = (func.__doc__ or "").strip().split("\n\n")[0].strip()
            schema.append({
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return schema

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
                tools=self._tools_schema,
                tool_choice="auto",
                temperature=1
            )

            reponse_message = response.choices[0].message

            print(f"LLM response: {reponse_message.content}")

            if reponse_message.tool_calls:
                tool_call_response = self._handle_tool_calls(reponse_message.tool_calls)
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

    def _handle_tool_calls(self, tool_calls) -> dict:
        for tool_call in tool_calls:
            function_name = tool_call.function.name

            function_args = json.loads(tool_call.function.arguments)

            print(f"Calling tool request: {function_name} with args: {function_args}")

            if function_name in self._available_tools:
                function_to_call = self._available_tools[function_name]
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
