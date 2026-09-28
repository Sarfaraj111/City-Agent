# 🏙️ City Agent

An AI-powered city assistant that fetches live weather and the latest news for any city, with human approval required before every tool call.

Built with LangChain, Google Gemini and Streamlit.

## Features

- 🌦️ **Live weather** – current conditions and temperature (°C) via the OpenWeatherMap API
- 📰 **Latest news** – top 3 recent stories about a city via the Tavily search API
- 🔐 **Human-in-the-loop** – the agent pauses before each tool call and asks you to approve or reject it
- 💬 **Chat interface** – Streamlit UI with conversation memory and a one-click "New chat" reset

## Demo Prompts

- "What's the weather in Tokyo?"
- "Give me the latest news in Paris."
- "Weather and news for Mumbai."

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Google Gemini (`langchain-google-genai`) |
| Agent framework | LangChain + LangGraph |
| Weather data | OpenWeatherMap API |
| News search | Tavily API |
| UI | Streamlit |

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/city-agent.git
cd city-agent
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API keys

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Get your keys here:
- Google Gemini: [Google AI Studio](https://aistudio.google.com/app/apikey)
- OpenWeatherMap: [openweathermap.org/api](https://openweathermap.org/api)
- Tavily: [tavily.com](https://tavily.com)

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## How It Works

1. You ask a question about a city.
2. The Gemini-powered agent decides which tool (`get_weather` or `get_news`) to call.
3. The agent **pauses** and shows the tool name and arguments in the UI.
4. You click **Approve** or **Reject**.
5. If approved, the tool runs and the agent replies with the result. If rejected, the agent continues without that data.

Approval is handled by LangChain's `HumanInTheLoopMiddleware` with an in-memory checkpointer, so the agent's state is preserved while it waits for your decision.

## Project Structure

```
city-agent/
├── app.py              # Streamlit app + agent + tools
├── requirements.txt    # Python dependencies
├── .env                # Your API keys (not committed)
├── .env.example        # Template for required keys
└── README.md
```
