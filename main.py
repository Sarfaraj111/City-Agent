from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("human", "Explain {topic} in simpler words")
])

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "Machine Learning"})

print(result)