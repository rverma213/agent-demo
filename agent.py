"""A minimal tool-using (ReAct-style) agent, built from scratch.

The whole idea fits in one loop:

    1. Send the conversation and the tool schemas to the model.
    2. If the model asks for tools (stop_reason == "tool_use"), run them,
       append the results to the conversation, and go back to step 1.
    3. If the model replies without asking for a tool (stop_reason == "end_turn"),
       that reply is the final answer.
    4. If the loop reaches MAX_STEPS, stop anyway. This is a guardrail, not success.

`run_agent` is a generator that yields an event for every step, so a UI can show
the agent's reasoning live instead of waiting for the final answer.

Run from the terminal:  python agent.py "What's the weather in Paris in Fahrenheit?"
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field

import anthropic

from tools import TOOL_SPECS, execute_tool

MODEL = os.environ.get("AGENT_MODEL", "claude-haiku-4-5-20251001")  # cheap and fast for a demo
MAX_STEPS = 6
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a helpful assistant that can use tools.

- Use get_weather for any question about current weather. Never guess the weather.
- Use calculator for any arithmetic, including unit conversions.
- If a tool returns an error, read it and try again with corrected input.
- When you have everything you need, give a short, direct answer.
- If a question can't be answered with your tools, say so briefly."""


# --- Events the agent yields -------------------------------------------------


@dataclass
class ToolCall:
    step: int
    name: str
    input: dict
    output: str
    is_error: bool


@dataclass
class FinalAnswer:
    text: str
    stop_reason: str  # "end_turn", "max_tokens", "max_steps", ...
    usage: dict = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.stop_reason == "end_turn"


AgentEvent = ToolCall | FinalAnswer


# --- The loop ----------------------------------------------------------------


def _text_of(content) -> str:
    return "".join(block.text for block in content if block.type == "text").strip()


def run_agent(
    goal: str,
    client: anthropic.Anthropic | None = None,
    max_steps: int = MAX_STEPS,
    model: str = MODEL,
) -> Iterator[AgentEvent]:
    """Run the agent on `goal`, yielding a ToolCall per tool run and one FinalAnswer."""
    client = client or anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    messages = [{"role": "user", "content": goal}]
    usage = {"input_tokens": 0, "output_tokens": 0, "model_calls": 0}

    for step in range(1, max_steps + 1):
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_SPECS,
            messages=messages,
        )
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens
        usage["model_calls"] += 1
        messages.append({"role": "assistant", "content": response.content})

        # Exit 1: the model answered without asking for a tool. This is the normal end.
        if response.stop_reason == "end_turn":
            yield FinalAnswer(_text_of(response.content), "end_turn", usage)
            return

        # Exit 2: anything else except a tool request (e.g. the reply hit max_tokens).
        # Report it honestly rather than passing off a truncated reply as an answer.
        if response.stop_reason != "tool_use":
            partial = _text_of(response.content)
            note = f"[Stopped early: {response.stop_reason}]"
            yield FinalAnswer(f"{partial}\n\n{note}".strip(), response.stop_reason, usage)
            return

        # Otherwise: run every tool the model asked for, then loop.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            output, is_error = execute_tool(block.name, block.input)
            yield ToolCall(step, block.name, block.input, output, is_error)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    # Exit 3: the safety net. The model kept asking for tools.
    yield FinalAnswer(
        f"I couldn't finish within {max_steps} steps, so I stopped.", "max_steps", usage
    )


def answer(goal: str, **kwargs) -> FinalAnswer:
    """Convenience wrapper: run the agent and return only the final answer."""
    final = None
    for event in run_agent(goal, **kwargs):
        if isinstance(event, FinalAnswer):
            final = event
    assert final is not None
    return final


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What's the weather in London, in Fahrenheit?"
    print(f"Q: {question}\n")
    for event in run_agent(question):
        if isinstance(event, ToolCall):
            flag = " (error)" if event.is_error else ""
            print(f"  [step {event.step}] {event.name}({event.input}){flag}\n    -> {event.output}")
        else:
            print(f"\nA: {event.text}")
            print(f"\n({event.stop_reason}; {event.usage})")
