import json
from unittest.mock import MagicMock

from agents.react_agent import ReactAgent
from agents._retriever import IdentityRetriever, Retriever


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


def _identity_for(tools):
    return IdentityRetriever([f.__name__ for f in tools])


class _StaticRetriever:
    """Test-only retriever that returns a fixed list of names per query."""

    def __init__(self, mapping: dict[str, list[str]] | None = None, names: list[str] | None = None,
                 error: Exception | None = None):
        self._mapping = mapping or {}
        self._names = names or []
        self._error = error
        self.calls: list[tuple[str, int]] = []

    def select(self, query: str, top_k: int) -> list[str]:
        self.calls.append((query, top_k))
        if self._error is not None:
            raise self._error
        if query in self._mapping:
            return self._mapping[query][:top_k]
        return self._names[:top_k]


class TestBuildToolsSchema:
    def test_empty_list_returns_empty_schema(self):
        agent = ReactAgent(MagicMock(), tools=[], filter_enabled=False)
        assert agent._tools_schema == []
        assert agent._available_tools == {}

    def test_basic_function_name_and_required_params(self):
        agent = ReactAgent(MagicMock(), tools=[_add], filter_enabled=False)
        fn = agent._tools_schema[0]["function"]
        assert fn["name"] == "_add"
        assert fn["parameters"]["required"] == ["a", "b"]
        assert fn["parameters"]["properties"]["a"]["type"] == "integer"
        assert fn["parameters"]["properties"]["b"]["type"] == "integer"

    def test_param_with_default_is_not_required(self):
        agent = ReactAgent(MagicMock(), tools=[_greet], filter_enabled=False)
        assert agent._tools_schema[0]["function"]["parameters"]["required"] == ["name"]

    def test_function_description_from_first_docstring_paragraph(self):
        agent = ReactAgent(MagicMock(), tools=[_with_doc], filter_enabled=False)
        assert agent._tools_schema[0]["function"]["description"] == "Do something useful."

    def test_param_description_from_args_block(self):
        agent = ReactAgent(MagicMock(), tools=[_with_doc], filter_enabled=False)
        props = agent._tools_schema[0]["function"]["parameters"]["properties"]
        assert props["a"]["description"] == "The input number."

    def test_no_docstring_yields_empty_strings(self):
        agent = ReactAgent(MagicMock(), tools=[_no_doc], filter_enabled=False)
        fn = agent._tools_schema[0]["function"]
        assert fn["description"] == ""
        assert fn["parameters"]["properties"]["a"]["description"] == ""

    def test_variadic_params_are_skipped(self):
        agent = ReactAgent(MagicMock(), tools=[_variadic], filter_enabled=False)
        params = agent._tools_schema[0]["function"]["parameters"]
        assert params["properties"] == {}
        assert params["required"] == []


class TestHandleToolCalls:
    def test_dispatches_to_matching_function(self):
        agent = ReactAgent(MagicMock(), tools=[_add, _greet], filter_enabled=False)
        tc = _make_tool_call("_add", {"a": 2, "b": 3})
        result = agent._handle_tool_calls([tc])
        assert result == {"tool_call_id": "call_1", "name": "_add", "content": 5}

    def test_unknown_function_returns_not_found(self):
        agent = ReactAgent(MagicMock(), tools=[_add], filter_enabled=False)
        tc = _make_tool_call("missing", {})
        result = agent._handle_tool_calls([tc])
        assert result["name"] == "missing"
        assert "not found" in result["content"].lower()
        assert result["tool_call_id"] == "call_1"


class TestCall:
    def test_returns_content_when_no_tool_calls(self):
        llm = MagicMock()
        llm.chat.completions.create.return_value = _mock_llm_response(content="hi there")
        agent = ReactAgent(llm=llm, tools=[_add], filter_enabled=False)

        assert agent.call("hello") == "hi there"

        kwargs = llm.chat.completions.create.call_args.kwargs
        assert kwargs["messages"][0]["role"] == "system"
        assert kwargs["messages"][-1] == {"role": "user", "content": "hello"}
        assert kwargs["tools"] == agent._tools_schema

    def test_history_inserted_between_system_and_user(self):
        llm = MagicMock()
        llm.chat.completions.create.return_value = _mock_llm_response(content="ok")
        agent = ReactAgent(llm=llm, tools=[_add], filter_enabled=False)

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

        agent = ReactAgent(llm=llm, tools=[_add], filter_enabled=False)
        result = agent.call("add 1 and 2")

        assert result == "the answer is 3"
        assert llm.chat.completions.create.call_count == 2

        second_msgs = llm.chat.completions.create.call_args_list[1].kwargs["messages"]
        tool_msgs = [m for m in second_msgs if isinstance(m, dict) and m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_42"
        assert tool_msgs[0]["name"] == "_add"
        assert tool_msgs[0]["content"] == "3"


class TestSelectTools:
    def test_full_schema_when_filter_disabled(self):
        retriever = _StaticRetriever(names=["_greet"])
        agent = ReactAgent(MagicMock(), tools=[_add, _greet], retriever=retriever, filter_enabled=False)
        assert agent._select_tools("anything") == agent._tools_schema
        assert retriever.calls == []

    def test_full_schema_when_top_k_is_zero(self):
        retriever = _StaticRetriever(names=["_greet"])
        agent = ReactAgent(MagicMock(), tools=[_add, _greet], retriever=retriever, top_k=0)
        assert agent._select_tools("anything") == agent._tools_schema
        assert retriever.calls == []

    def test_full_schema_when_no_tools_registered(self):
        retriever = _StaticRetriever(names=[])
        agent = ReactAgent(MagicMock(), tools=[], retriever=retriever)
        assert agent._select_tools("anything") == []

    def test_filter_returns_only_selected_names(self):
        retriever = _StaticRetriever(mapping={"sum these": ["_add"]})
        agent = ReactAgent(
            MagicMock(), tools=[_add, _greet], retriever=retriever, top_k=5
        )
        selected = agent._select_tools("sum these")
        names = [s["function"]["name"] for s in selected]
        assert names == ["_add"]

    def test_filter_respects_top_k(self):
        retriever = _StaticRetriever(names=["_add", "_greet", "_with_doc", "_no_doc"])
        agent = ReactAgent(
            MagicMock(),
            tools=[_add, _greet, _with_doc, _no_doc],
            retriever=retriever,
            top_k=2,
        )
        selected = agent._select_tools("anything")
        assert len(selected) == 2
        assert retriever.calls == [("anything", 2)]

    def test_filter_ignores_unknown_names(self):
        retriever = _StaticRetriever(names=["_add", "_does_not_exist"])
        agent = ReactAgent(
            MagicMock(), tools=[_add, _greet], retriever=retriever, top_k=5
        )
        selected = agent._select_tools("anything")
        names = [s["function"]["name"] for s in selected]
        assert names == ["_add"]

    def test_filter_falls_back_on_retriever_error(self):
        retriever = _StaticRetriever(error=RuntimeError("ollama down"))
        agent = ReactAgent(
            MagicMock(), tools=[_add, _greet], retriever=retriever, top_k=5
        )
        assert agent._select_tools("anything") == agent._tools_schema

    def test_filter_falls_back_on_empty_result(self):
        retriever = _StaticRetriever(names=[])
        agent = ReactAgent(
            MagicMock(), tools=[_add, _greet], retriever=retriever, top_k=5
        )
        assert agent._select_tools("anything") == agent._tools_schema

    def test_call_sends_filtered_tools_to_llm(self):
        retriever = _StaticRetriever(mapping={"weather in Tokyo": ["_greet"]})
        llm = MagicMock()
        llm.chat.completions.create.return_value = _mock_llm_response(content="hi")
        agent = ReactAgent(
            llm=llm, tools=[_add, _greet], retriever=retriever, top_k=5
        )

        agent.call("weather in Tokyo")

        sent = llm.chat.completions.create.call_args.kwargs["tools"]
        sent_names = [s["function"]["name"] for s in sent]
        assert sent_names == ["_greet"]
        assert retriever.calls == [("weather in Tokyo", 5)]


class TestRetrieverImplementations:
    def test_identity_retriever_returns_all_in_order(self):
        r = IdentityRetriever(["_a", "_b", "_c"])
        assert r.select("anything", top_k=10) == ["_a", "_b", "_c"]

    def test_identity_retriever_respects_top_k(self):
        r = IdentityRetriever(["_a", "_b", "_c"])
        assert r.select("anything", top_k=2) == ["_a", "_b"]

    def test_identity_retriever_handles_zero_top_k(self):
        r = IdentityRetriever(["_a", "_b"])
        assert r.select("anything", top_k=0) == []
