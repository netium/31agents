"""LLM-as-judge tests for ReactAgent.

These tests require a running Ollama instance with the configured model
(see OLLAMA_BASE_URL and OLLAMA_LLM_MODEL in .env). They are marked with
`@pytest.mark.llm` and skipped automatically if Ollama is unreachable.

Run with:
    uv run --group dev pytest -m llm

Run only the fast unit tests:
    uv run --group dev pytest -m "not llm"
"""

import json
import os
import re
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from openai import OpenAI

from agents._retriever import IdentityRetriever
from agents.react_agent import ReactAgent
from react import add, get_weather

pytestmark = pytest.mark.llm


def _ollama_reachable() -> bool:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    root = re.sub(r"/v1/?$", "", base)
    try:
        with urlopen(f"{root}/api/tags", timeout=2) as r:
            return 200 <= r.status < 300
    except (URLError, OSError):
        return False


def _make_llm() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
    )


def _make_agent() -> ReactAgent:
    return ReactAgent(
        llm=_make_llm(),
        tools=[add, get_weather],
        retriever=IdentityRetriever(["add", "get_weather"]),
    )


@pytest.fixture(autouse=True)
def _require_ollama():
    if not _ollama_reachable():
        pytest.skip("Ollama not reachable at " + os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"))


JUDGE_SYSTEM = """You are an impartial judge evaluating a ReAct agent's response.
You will be given a user question, a rubric, the agent's final answer, and a trace of any tool calls the agent made.
Respond with a single JSON object on one line: {"pass": true|false, "reasoning": "<one-sentence explanation>"}.
Do not include any other text."""


def _judge(question: str, rubric: str, answer: str, trace: list[dict]) -> tuple[bool, str]:
    if trace:
        trace_str = "\n".join(f"- {t['name']}({t['args']}) -> {t['result']!r}" for t in trace)
    else:
        trace_str = "(no tool calls)"
    user_prompt = (
        f"User question: {question}\n\n"
        f"Rubric:\n{rubric}\n\n"
        f"Tool calls:\n{trace_str}\n\n"
        f"Final answer: {answer}\n"
    )
    llm = _make_llm()
    response = llm.chat.completions.create(
        model=os.getenv("OLLAMA_LLM_MODEL", "qwen3.6:35b"),
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    text = response.choices[0].message.content.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            return False, f"Judge returned non-JSON: {text[:200]}"
    return bool(data.get("pass")), str(data.get("reasoning", ""))


def _run_with_trace(agent: ReactAgent, prompt: str) -> tuple[str, list[dict]]:
    trace: list[dict] = []
    original = agent._handle_tool_calls

    def spy(tool_calls):
        result = original(tool_calls)
        for tc in tool_calls:
            trace.append({
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments),
                "result": result["content"],
            })
        return result

    agent._handle_tool_calls = spy
    answer = agent.call(prompt)
    return answer, trace


def test_add_tool_used_correctly():
    agent = _make_agent()
    answer, trace = _run_with_trace(agent, "What is 7 + 13? Use the add tool.")
    rubric = (
        "The agent must have called the 'add' tool with a=7 and b=13, "
        "and the final answer must contain the number 20."
    )
    passed, reason = _judge("What is 7 + 13?", rubric, answer, trace)
    assert passed, f"Judge failed: {reason}\nAnswer: {answer}\nTrace: {trace}"


def test_weather_tool_used_correctly():
    agent = _make_agent()
    answer, trace = _run_with_trace(agent, "What's the weather in Tokyo? Use the get_weather tool.")
    rubric = (
        "The agent must have called the 'get_weather' tool with location='Tokyo', "
        "and the final answer must mention the weather (sunny / temperature)."
    )
    passed, reason = _judge("What's the weather in Tokyo?", rubric, answer, trace)
    assert passed, f"Judge failed: {reason}\nAnswer: {answer}\nTrace: {trace}"


def test_no_tool_for_simple_greeting():
    agent = _make_agent()
    answer, trace = _run_with_trace(agent, "Just say hello back to me, no tools needed.")
    rubric = (
        "The agent must NOT have called any tool. "
        "The final answer must be a friendly greeting."
    )
    passed, reason = _judge("Just say hello back to me.", rubric, answer, trace)
    assert passed, f"Judge failed: {reason}\nAnswer: {answer}\nTrace: {trace}"
