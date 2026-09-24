"""Streamlit web UI for the agent. Run locally with:  streamlit run app.py"""

from __future__ import annotations

import os

import streamlit as st

st.set_page_config(page_title="Tool-Using Agent Demo", page_icon="🤖", layout="centered")

# --- API key: Streamlit secrets in the cloud, environment variable locally ---
try:
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
except Exception:  # no secrets.toml at all
    api_key = None
api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    st.error(
        "No API key configured. Add `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` "
        "(see `secrets.toml.example`) or set it as an environment variable."
    )
    st.stop()
os.environ["ANTHROPIC_API_KEY"] = api_key

from agent import MAX_STEPS, MODEL, FinalAnswer, ToolCall, run_agent  # noqa: E402

# --- Demo guardrails: protect the owner's API budget ---
MAX_QUESTIONS_PER_SESSION = 5
MAX_QUESTION_CHARS = 500

EXAMPLES = [
    "What's the weather in Paris right now, in Fahrenheit?",
    "Is it warmer in Tokyo or Sydney at the moment? By how much?",
    "What is (17 * 23) + 2**10?",
    "If I split £1,284.50 between 7 people, how much each?",
]

st.session_state.setdefault("history", [])  # list of (question, events)
st.session_state.setdefault("count", 0)

# --- Sidebar ---
with st.sidebar:
    st.header("About")
    st.markdown(
        "An **AI agent built from scratch** in Python, with no agent framework. "
        "The model decides which tools to call and in what order; the code runs "
        "them and feeds the results back until the model gives a final answer."
    )
    st.markdown(
        "**Tools**\n- 🌦️ `get_weather`: live data from Open-Meteo\n"
        "- 🧮 `calculator`: safe arithmetic parser (no `eval`)"
    )
    st.markdown(f"**Model:** `{MODEL}`  \n**Step limit:** {MAX_STEPS}")
    remaining = MAX_QUESTIONS_PER_SESSION - st.session_state.count
    st.progress(
        max(remaining, 0) / MAX_QUESTIONS_PER_SESSION,
        text=f"Questions left this session: {max(remaining, 0)}",
    )
    st.markdown("[Source code on GitHub](https://github.com/rverma213/agent-demo)")

st.title("🤖 Tool-Using Agent")
st.caption("Ask a question and watch the agent choose its tools, step by step.")


def render(events: list) -> None:
    for event in events:
        if isinstance(event, ToolCall):
            icon = "⚠️" if event.is_error else "🔧"
            args = ", ".join(f"{k}={v!r}" for k, v in event.input.items())
            with st.status(
                f"{icon} Step {event.step}: `{event.name}({args})`",
                state="error" if event.is_error else "complete",
            ):
                st.code(event.output, language=None)
        elif isinstance(event, FinalAnswer):
            st.markdown(event.text)
            u = event.usage
            st.caption(
                f"Stopped because: `{event.stop_reason}` · {u.get('model_calls', 0)} model calls · "
                f"{u.get('input_tokens', 0):,} input / {u.get('output_tokens', 0):,} output tokens"
            )


# --- Replay earlier turns (Streamlit reruns the script on every interaction) ---
for question, events in st.session_state.history:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        render(events)

# --- Example buttons, shown until the first question ---
clicked = None
if not st.session_state.history:
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example):
            clicked = example

goal = st.chat_input("Ask the agent something...") or clicked

if goal:
    goal = goal.strip()
    if st.session_state.count >= MAX_QUESTIONS_PER_SESSION:
        st.warning("You've reached the demo limit for this session. Thanks for trying it!")
        st.stop()
    if len(goal) > MAX_QUESTION_CHARS:
        st.warning(f"Please keep questions under {MAX_QUESTION_CHARS} characters.")
        st.stop()

    st.session_state.count += 1
    st.chat_message("user").write(goal)
    events: list = []
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                for event in run_agent(goal):
                    events.append(event)
                    render([event])
        except Exception as exc:  # show a friendly error, not a stack trace
            st.error(f"Something went wrong talking to the model: {exc.__class__.__name__}")
            events.append(FinalAnswer("Sorry, something went wrong. Please try again.", "error"))
    st.session_state.history.append((goal, events))
    st.rerun()
