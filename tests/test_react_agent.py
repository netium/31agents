import json
from unittest.mock import MagicMock

from agents.react_agent import ReactAgent


def _add(a: int, b: int) -> int:
    return a + b


def _greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"


def _no_doc(a: int) -> int:
    return a


def _with_doc(a: int) -> int:
    """Do something useful.

    Args:
        a: The input number.
    """
    return a


def _variadic(*args, **kwargs):
    return args, kwargs


def _make_tool_call(name, args, call_id="call_1"):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def _mock_llm_response(content=None, tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    return MagicMock(choices=[MagicMock(message=msg)])


class TestBuildToolsSchema:
    def test_empty_list_returns_empty_schema(self):
        agent = ReactAgent(MagicMock(), tools=[])
        assert agent._tools_schema == []
        assert agent._available_tools == {}

    def test_basic_function_name_and_required_params(self):
        agent = ReactAgent(MagicMock(), tools=[_add])
        fn = agent._tools_schema[0]["function"]
        assert fn["name"] == "_add"
        assert fn["parameters"]["required"] == ["a", "b"]
        assert fn["parameters"]["properties"]["a"]["type"] == "integer"
        assert fn["parameters"]["properties"]["b"]["type"] == "integer"

    def test_param_with_default_is_not_required(self):
        agent = ReactAgent(MagicMock(), tools=[_greet])
        assert agent._tools_schema[0]["function"]["parameters"]["required"] == ["name"]

    def test_function_description_from_first_docstring_paragraph(self):
        agent = ReactAgent(MagicMock(), tools=[_with_doc])
        assert agent._tools_schema[0]["function"]["description"] == "Do something useful."

    def test_param_description_from_args_block(self):
        agent = ReactAgent(MagicMock(), tools=[_with_doc])
        props = agent._tools_schema[0]["function"]["parameters"]["properties"]
        assert props["a"]["description"] == "The input number."

    def test_no_docstring_yields_empty_strings(self):
        agent = ReactAgent(MagicMock(), tools=[_no_doc])
        fn = agent._tools_schema[0]["function"]
        assert fn["description"] == ""
        assert fn["parameters"]["properties"]["a"]["description"] == ""

    def test_variadic_params_are_skipped(self):
        agent = ReactAgent(MagicMock(), tools=[_variadic])
        params = agent._tools_schema[0]["function"]["parameters"]
        assert params["properties"] == {}
        assert params["required"] == []


class TestHandleToolCalls:
    def test_dispatches_to_matching_function(self):
        agent = ReactAgent(MagicMock(), tools=[_add, _greet])
        tc = _make_tool_call("_add", {"a": 2, "b": 3})
        result = agent._handle_tool_calls([tc])
        assert result == {"tool_call_id": "call_1", "name": "_add", "content": 5}

    def test_unknown_function_returns_not_found(self):
        agent = ReactAgent(MagicMock(), tools=[_add])
        tc = _make_tool_call("missing", {})
        result = agent._handle_tool_calls([tc])
        assert result["name"] == "missing"
        assert "not found" in result["content"].lower()
        assert result["tool_call_id"] == "call_1"


class TestCall:
    def test_returns_content_when_no_tool_calls(self):
        llm = MagicMock()
        llm.chat.completions.create.return_value = _mock_llm_response(content="hi there")
        agent = ReactAgent(llm=llm, tools=[_add])

        assert agent.call("hello") == "hi there"

        kwargs = llm.chat.completions.create.call_args.kwargs
        assert kwargs["messages"][0]["role"] == "system"
        assert kwargs["messages"][-1] == {"role": "user", "content": "hello"}
        assert kwargs["tools"] == agent._tools_schema

    def test_history_inserted_between_system_and_user(self):
        llm = MagicMock()
        llm.chat.completions.create.return_value = _mock_llm_response(content="ok")
        agent = ReactAgent(llm=llm, tools=[_add])

        history = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ]
        agent.call("now", history=history)

        msgs = llm.chat.completions.create.call_args.kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1:3] == history
        assert msgs[-1] == {"role": "user", "content": "now"}

    def test_tool_call_loop_returns_final_content(self):
        tool_tc = _make_tool_call("_add", {"a": 1, "b": 2}, call_id="call_42")
        tool_response = _mock_llm_response(content=None, tool_calls=[tool_tc])
        final_response = _mock_llm_response(content="the answer is 3")

        llm = MagicMock()
        llm.chat.completions.create.side_effect = [tool_response, final_response]

        agent = ReactAgent(llm=llm, tools=[_add])
        result = agent.call("add 1 and 2")

        assert result == "the answer is 3"
        assert llm.chat.completions.create.call_count == 2

        second_msgs = llm.chat.completions.create.call_args_list[1].kwargs["messages"]
        tool_msgs = [m for m in second_msgs if isinstance(m, dict) and m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_42"
        assert tool_msgs[0]["name"] == "_add"
        assert tool_msgs[0]["content"] == "3"
