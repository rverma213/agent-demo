"""Tests for the agent loop, using a scripted fake model instead of the real API.

These check the loop's *control flow*: that it runs the tools the model asks
for, feeds results back, and stops for the right reason.
"""

from types import SimpleNamespace as NS

from agent import FinalAnswer, ToolCall, answer, run_agent


def text(t):
    return NS(type="text", text=t)


def tool_use(id_, name, input_):
    return NS(type="tool_use", id=id_, name=name, input=input_)


def reply(stop_reason, *content):
    return NS(
        stop_reason=stop_reason, content=list(content), usage=NS(input_tokens=100, output_tokens=20)
    )


class FakeClient:
    """Returns pre-scripted responses and records every request it receives."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        # Snapshot the message list: the agent keeps appending to the same list object.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


def test_answers_directly_without_tools():
    client = FakeClient(reply("end_turn", text("Hello!")))
    events = list(run_agent("Hi", client=client))
    assert events == [FinalAnswer("Hello!", "end_turn", events[0].usage)]
    assert events[0].completed
    assert len(client.requests) == 1


def test_multi_step_tool_use_then_answer():
    client = FakeClient(
        reply(
            "tool_use",
            text("Converting."),
            tool_use("t1", "calculator", {"expression": "14*9/5+32"}),
        ),
        reply("end_turn", text("It's 57.2°F.")),
    )
    events = list(run_agent("14C in F?", client=client))

    assert isinstance(events[0], ToolCall)
    assert (events[0].name, events[0].output, events[0].is_error) == ("calculator", "57.2", False)
    assert events[-1].text == "It's 57.2°F."
    assert events[-1].usage == {"input_tokens": 200, "output_tokens": 40, "model_calls": 2}

    # The tool result must be sent back to the model, linked to the right tool call.
    second_request = client.requests[1]["messages"]
    result_block = second_request[-1]["content"][0]
    assert result_block == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": "57.2",
        "is_error": False,
    }


def test_parallel_tool_calls_in_one_step():
    client = FakeClient(
        reply(
            "tool_use",
            tool_use("a", "calculator", {"expression": "1+1"}),
            tool_use("b", "calculator", {"expression": "2+2"}),
        ),
        reply("end_turn", text("2 and 4")),
    )
    events = list(run_agent("both", client=client))
    assert [e.output for e in events if isinstance(e, ToolCall)] == ["2", "4"]
    assert len(client.requests[1]["messages"][-1]["content"]) == 2


def test_tool_errors_are_fed_back_so_the_model_can_recover():
    client = FakeClient(
        reply("tool_use", tool_use("t1", "calculator", {"expression": "import os"})),
        reply("tool_use", tool_use("t2", "calculator", {"expression": "2*3"})),
        reply("end_turn", text("6")),
    )
    events = list(run_agent("x", client=client))
    assert events[0].is_error
    assert client.requests[1]["messages"][-1]["content"][0]["is_error"] is True
    assert events[-1].text == "6"


def test_step_limit_stops_a_runaway_loop():
    looping = [
        reply("tool_use", tool_use(f"t{i}", "calculator", {"expression": "1"})) for i in range(10)
    ]
    client = FakeClient(*looping)
    final = answer("loop forever", client=client, max_steps=3)
    assert final.stop_reason == "max_steps"
    assert not final.completed
    assert len(client.requests) == 3


def test_truncated_reply_is_not_passed_off_as_complete():
    client = FakeClient(reply("max_tokens", text("The answer is a long")))
    final = answer("x", client=client)
    assert final.stop_reason == "max_tokens"
    assert not final.completed
    assert "Stopped early: max_tokens" in final.text
