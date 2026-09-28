import os
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from tavily import TavilyClient

load_dotenv()

# ----------------------------------------------------------------------------
# Tools (unchanged)
# ----------------------------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city."""
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": API_KEY, "units": "metric"}

    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code != 200:
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    return f"Weather in {city}: {desc}, {temp}°C"


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""
    response = tavily_client.search(
        query=f"latest news in {city}", search_depth="basic", max_results=3
    )
    results = response.get("results", [])
    if not results:
        return f"No news found for {city}"

    news_list = []
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        news_list.append(f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}...")

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


# ----------------------------------------------------------------------------
# Agent (cached so the checkpointer/memory survives Streamlit reruns)
#
# NOTE: The original `input()`-based approval middleware can't work in Streamlit
# (it blocks the script). HumanInTheLoopMiddleware pauses the agent instead,
# and we resume it when the user clicks Approve / Reject.
# ----------------------------------------------------------------------------
@st.cache_resource
def get_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    return create_agent(
        llm,
        tools=[get_news, get_weather],
        system_prompt="you are a helpful city assistant.",
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"get_news": True, "get_weather": True}
            )
        ],
        checkpointer=InMemorySaver(),
    )


agent = get_agent()

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def text_of(content) -> str:
    """Gemini may return a list of content blocks; flatten to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


def handle_result(result: dict):
    """Either store a pending approval request or append the final answer."""
    interrupts = result.get("__interrupt__")
    if interrupts:
        st.session_state.pending = interrupts[0].value["action_requests"]
    else:
        st.session_state.pending = None
        st.session_state.messages.append(
            {"role": "assistant", "content": text_of(result["messages"][-1].content)}
        )


def run_agent(payload):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.spinner("Thinking..."):
        try:
            handle_result(agent.invoke(payload, config=config))
        except Exception as e:
            st.session_state.pending = None
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ Error: {e}"}
            )


# ----------------------------------------------------------------------------
# Page + state
# ----------------------------------------------------------------------------
st.set_page_config(page_title="City Agent", page_icon="🏙️")
st.title("🏙️ City Agent")
st.caption("Ask about the weather or latest news in any city. "
           "You'll be asked to approve each tool call.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "pending" not in st.session_state:
    st.session_state.pending = None

with st.sidebar:
    st.header("Options")
    if st.button("🗑️ New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    st.markdown("**Try:**")
    st.markdown("- What's the weather in Kolkata?\n- Any news in London?\n"
                "- Weather and news for Tokyo")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------------------------------------------------------
# Human approval panel
# ----------------------------------------------------------------------------
if st.session_state.pending:
    actions = st.session_state.pending
    with st.chat_message("assistant"):
        st.warning("🔐 The agent wants to call the following tool(s):")
        for a in actions:
            st.code(f"{a['name']}({a['args']})", language="python")

        col1, col2 = st.columns(2)
        approve = col1.button("✅ Approve", use_container_width=True, type="primary")
        reject = col2.button("❌ Reject", use_container_width=True)

    if approve or reject:
        names = ", ".join(a["name"] for a in actions)
        if approve:
            decisions = [{"type": "approve"} for _ in actions]
            note = f"✅ Approved: `{names}`"
        else:
            decisions = [
                {"type": "reject", "message": "Tool call denied by the user"}
                for _ in actions
            ]
            note = f"❌ Rejected: `{names}`"

        st.session_state.messages.append({"role": "assistant", "content": note})
        run_agent(Command(resume={"decisions": decisions}))
        st.rerun()

# ----------------------------------------------------------------------------
# Chat input (disabled while waiting for approval)
# ----------------------------------------------------------------------------
prompt = st.chat_input(
    "Ask about a city...", disabled=st.session_state.pending is not None
)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    run_agent({"messages": [{"role": "user", "content": prompt}]})
    st.rerun()