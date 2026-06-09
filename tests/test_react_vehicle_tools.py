"""Tests for the vehicle-control tools defined in `react.py`.

Two tiers:

* `TestVehicleToolFunctions` — **unit tests** for the pure functions (no agent,
  no LLM). Deterministic, fast.
* `TestVehicleToolsWithReactAgent` — **system tests** that exercise the tools
  through the full `ReactAgent.call` loop with a mocked LLM, verifying that
  the tools are registered, exposed via the OpenAI-compatible tool schema, and
  that the agent can dispatch a tool call to them and surface the result back
  to the LLM.
"""

import json
from unittest.mock import MagicMock

from agents.react_agent import ReactAgent
from react import (
    airconditioner_control,
    headlight_control,
    seat_heating_control,
    seat_massage_control,
    sunroof_control,
    volume_control,
    window_control,
)

ALL_VEHICLE_TOOLS = [
    headlight_control,
    window_control,
    airconditioner_control,
    seat_heating_control,
    seat_massage_control,
    sunroof_control,
    volume_control,
]
ALL_VEHICLE_TOOL_NAMES = {f.__name__ for f in ALL_VEHICLE_TOOLS}


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


class TestVehicleToolFunctions:
    """Direct unit tests for the seven vehicle-control functions."""

    # --- headlight_control ---

    def test_headlight_on_uses_defaults(self):
        assert headlight_control("on") == "Headlights turned on (low beam, 100% intensity)."

    def test_headlight_on_with_custom_mode_and_intensity(self):
        assert (
            headlight_control("on", mode="high", intensity=80)
            == "Headlights turned on (high beam, 80% intensity)."
        )

    def test_headlight_on_with_auto_mode(self):
        assert (
            headlight_control("on", mode="auto", intensity=50)
            == "Headlights turned on (auto beam, 50% intensity)."
        )

    def test_headlight_off_ignores_mode_and_intensity(self):
        assert (
            headlight_control("off", mode="high", intensity=80)
            == "Headlights turned off."
        )

    # --- window_control ---

    def test_window_default_seat_is_driver(self):
        assert window_control(50) == "driver window set to 50% open."

    def test_window_named_seat(self):
        assert window_control(0, window="rear_right") == "rear_right window set to 0% open."
        assert window_control(100, window="all") == "all window set to 100% open."

    # --- airconditioner_control ---

    def test_airconditioner_on_uses_defaults(self):
        assert (
            airconditioner_control("on")
            == "Air conditioner turned on — target 22°C, mode: auto."
        )

    def test_airconditioner_on_with_custom_temperature_and_mode(self):
        assert (
            airconditioner_control("on", temperature=18, mode="cool")
            == "Air conditioner turned on — target 18°C, mode: cool."
        )

    def test_airconditioner_off_ignores_optional_params(self):
        assert (
            airconditioner_control("off", temperature=18, mode="heat")
            == "Air conditioner turned off."
        )

    # --- seat_heating_control ---

    def test_seat_heating_driver_high(self):
        assert seat_heating_control("driver", "high") == "Driver seat heating set to high."

    def test_seat_heating_passenger_off(self):
        assert (
            seat_heating_control("passenger", "off")
            == "Passenger seat heating set to off."
        )

    # --- seat_massage_control ---

    def test_seat_massage_on_default_mode(self):
        assert (
            seat_massage_control("driver", "on")
            == "Driver seat massage turned on (wave mode)."
        )

    def test_seat_massage_on_with_pulse_mode(self):
        assert (
            seat_massage_control("passenger", "on", mode="pulse")
            == "Passenger seat massage turned on (pulse mode)."
        )

    def test_seat_massage_off_ignores_mode(self):
        assert (
            seat_massage_control("driver", "off", mode="pulse")
            == "Driver seat massage turned off."
        )

    # --- sunroof_control ---

    def test_sunroof_fully_open(self):
        assert sunroof_control(100) == "Sunroof set to 100% open."

    def test_sunroof_fully_closed(self):
        assert sunroof_control(0) == "Sunroof set to 0% open."

    def test_sunroof_partial(self):
        assert sunroof_control(40) == "Sunroof set to 40% open."

    # --- volume_control ---

    def test_volume_set(self):
        assert volume_control("set", 40) == "Volume set to 40%."

    def test_volume_increase(self):
        assert volume_control("increase", 10) == "Volume increased by 10%."

    def test_volume_decrease(self):
        assert volume_control("decrease", 5) == "Volume decreased by 5%."

    def test_volume_mute(self):
        assert volume_control("mute") == "Audio muted."

    def test_volume_unknown_action(self):
        assert volume_control("explode") == "Unknown volume action: explode."


class TestVehicleToolsWithReactAgent:
    """System tests: wire the vehicle tools through the full ReactAgent call loop."""

    def test_all_vehicle_tools_are_registered(self):
        """All seven vehicle tools appear in the agent's schema and dispatcher."""
        llm = MagicMock()
        agent = ReactAgent(llm=llm, tools=ALL_VEHICLE_TOOLS, filter_enabled=False)

        schema_names = {s["function"]["name"] for s in agent._tools_schema}
        assert schema_names == ALL_VEHICLE_TOOL_NAMES
        assert set(agent._available_tools) == ALL_VEHICLE_TOOL_NAMES

    def test_required_params_in_schema(self):
        """The schema flags the same required params as the Python signatures."""
        llm = MagicMock()
        agent = ReactAgent(llm=llm, tools=ALL_VEHICLE_TOOLS, filter_enabled=False)
        by_name = {s["function"]["name"]: s for s in agent._tools_schema}

        assert by_name["headlight_control"]["function"]["parameters"]["required"] == ["action"]
        assert by_name["window_control"]["function"]["parameters"]["required"] == ["position"]
        assert by_name["airconditioner_control"]["function"]["parameters"]["required"] == ["action"]
        assert by_name["seat_heating_control"]["function"]["parameters"]["required"] == ["seat", "level"]
        assert by_name["seat_massage_control"]["function"]["parameters"]["required"] == ["seat", "action"]
        assert by_name["sunroof_control"]["function"]["parameters"]["required"] == ["position"]
        assert by_name["volume_control"]["function"]["parameters"]["required"] == ["action"]

    def test_optional_params_are_not_required(self):
        """Params with defaults should not be in the `required` list."""
        llm = MagicMock()
        agent = ReactAgent(llm=llm, tools=ALL_VEHICLE_TOOLS, filter_enabled=False)
        by_name = {s["function"]["name"]: s for s in agent._tools_schema}

        # headlight_control: mode and intensity have defaults
        headlight_required = by_name["headlight_control"]["function"]["parameters"]["required"]
        assert "mode" not in headlight_required
        assert "intensity" not in headlight_required

        # airconditioner_control: temperature and mode have defaults
        ac_required = by_name["airconditioner_control"]["function"]["parameters"]["required"]
        assert "temperature" not in ac_required
        assert "mode" not in ac_required

        # window_control: window has a default
        assert "window" not in by_name["window_control"]["function"]["parameters"]["required"]

        # seat_massage_control: mode has a default
        assert "mode" not in by_name["seat_massage_control"]["function"]["parameters"]["required"]

        # volume_control: level has a default
        assert "level" not in by_name["volume_control"]["function"]["parameters"]["required"]

    def test_schema_param_types(self):
        """Verify Python type hints map to the expected JSON-schema types."""
        llm = MagicMock()
        agent = ReactAgent(llm=llm, tools=ALL_VEHICLE_TOOLS, filter_enabled=False)
        by_name = {s["function"]["name"]: s for s in agent._tools_schema}

        assert by_name["headlight_control"]["function"]["parameters"]["properties"]["action"]["type"] == "string"
        assert by_name["headlight_control"]["function"]["parameters"]["properties"]["intensity"]["type"] == "integer"
        assert by_name["window_control"]["function"]["parameters"]["properties"]["position"]["type"] == "integer"
        assert by_name["airconditioner_control"]["function"]["parameters"]["properties"]["temperature"]["type"] == "integer"
        assert by_name["sunroof_control"]["function"]["parameters"]["properties"]["position"]["type"] == "integer"

    def test_call_dispatches_to_window_control(self):
        """End-to-end: LLM emits a tool call → agent runs window_control → result returned."""
        tool_tc = _make_tool_call(
            "window_control",
            {"position": 60, "window": "passenger"},
            call_id="call_w",
        )
        tool_response = _mock_llm_response(content=None, tool_calls=[tool_tc])
        final_response = _mock_llm_response(content="Passenger window is now 60% open.")

        llm = MagicMock()
        llm.chat.completions.create.side_effect = [tool_response, final_response]

        agent = ReactAgent(llm=llm, tools=[window_control], filter_enabled=False)

        result = agent.call("open the passenger window to 60%")

        assert result == "Passenger window is now 60% open."
        assert llm.chat.completions.create.call_count == 2

        # The tool message sent back to the LLM should carry the function's return value.
        second_msgs = llm.chat.completions.create.call_args_list[1].kwargs["messages"]
        tool_msgs = [m for m in second_msgs if isinstance(m, dict) and m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_w"
        assert tool_msgs[0]["name"] == "window_control"
        assert tool_msgs[0]["content"] == "passenger window set to 60% open."

    def test_call_dispatches_to_seat_heating(self):
        tool_tc = _make_tool_call(
            "seat_heating_control",
            {"seat": "driver", "level": "high"},
            call_id="call_h",
        )
        tool_response = _mock_llm_response(content=None, tool_calls=[tool_tc])
        final_response = _mock_llm_response(content="Driver seat heating is now on high.")

        llm = MagicMock()
        llm.chat.completions.create.side_effect = [tool_response, final_response]

        agent = ReactAgent(llm=llm, tools=[seat_heating_control], filter_enabled=False)

        assert (
            agent.call("turn on driver seat heating to high")
            == "Driver seat heating is now on high."
        )

    def test_call_dispatches_to_volume_control_mute(self):
        tool_tc = _make_tool_call("volume_control", {"action": "mute"}, call_id="call_v")
        tool_response = _mock_llm_response(content=None, tool_calls=[tool_tc])
        final_response = _mock_llm_response(content="Audio muted.")

        llm = MagicMock()
        llm.chat.completions.create.side_effect = [tool_response, final_response]

        agent = ReactAgent(llm=llm, tools=[volume_control], filter_enabled=False)

        assert agent.call("mute the audio") == "Audio muted."

    def test_call_dispatches_to_headlight_then_sunroof(self):
        """Two-turn ReAct loop using different vehicle tools across iterations."""
        headlight_tc = _make_tool_call(
            "headlight_control",
            {"action": "on", "mode": "high", "intensity": 90},
            call_id="call_h1",
        )
        sunroof_tc = _make_tool_call("sunroof_control", {"position": 50}, call_id="call_s")

        turn1 = _mock_llm_response(content=None, tool_calls=[headlight_tc])
        turn2 = _mock_llm_response(content=None, tool_calls=[sunroof_tc])
        final = _mock_llm_response(content="Headlights on (high, 90%); sunroof half open.")

        llm = MagicMock()
        llm.chat.completions.create.side_effect = [turn1, turn2, final]

        agent = ReactAgent(
            llm=llm,
            tools=[headlight_control, sunroof_control],
            filter_enabled=False,
        )

        assert (
            agent.call("turn on the high beams and open the sunroof halfway")
            == "Headlights on (high, 90%); sunroof half open."
        )
        assert llm.chat.completions.create.call_count == 3

    def test_tools_sent_to_llm_match_agent_schema(self):
        """The tools kwarg forwarded to the OpenAI client equals the agent's schema."""
        llm = MagicMock()
        llm.chat.completions.create.return_value = _mock_llm_response(content="ok")

        agent = ReactAgent(llm=llm, tools=ALL_VEHICLE_TOOLS, filter_enabled=False)
        agent.call("hi")

        kwargs = llm.chat.completions.create.call_args.kwargs
        assert kwargs["tools"] == agent._tools_schema
