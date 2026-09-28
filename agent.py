import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent
from rich import print
from langchain.agents.middleware import wrap_tool_call


load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city."""

    API_KEY = os.getenv("OPENWEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        data = response.json()
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    data = response.json()

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""
    
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )
    
    results = response.get("results", [])
    
    if not results:
        return f"No news found for {city}"
    
    news_list = []
    
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        
        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )
    
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)

llm = ChatGoogleGenerativeAI(model= "gemini-3.6-flash")

@wrap_tool_call
def human_approval(request , handler):
    """Ask Human approval before tool calling"""
    tool_name= request.tool_call["name"]
    confirm=input(f"Agent wants to call'{tool_name}'.approve(yes/no)")
    if confirm.lower() != "yes":
        return ToolMessage(
            content="Tool call denied by the user",
            tool_call_id=request.tool_call["id"]        
        )
    return handler(request)

agent = create_agent(
    llm,
    tools = [get_news,get_weather],
    system_prompt="you are a helpful city assistant.",
    middleware=[human_approval]
)

print("City Agent! | Type exit to exit")

while True:
    user_input = input("You : ")
    if user_input.lower()=="exit":
        break
    result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]}
    )
    print("bot : ", result['messages'][-1].content )