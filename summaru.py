from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


search_tool = TavilySearch(max_results=5)


prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Summarize the news in bullet points.

{news}
"""
)


model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


chain = prompt | model | StrOutputParser()


news_result = search_tool.invoke({
    "query": "LATEST AI NEWS 2026"
})


result = chain.invoke({
    "news": news_result
})


print(result)