from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


short_template = ChatPromptTemplate.from_template(
    "Explain the {topic} in 2-3 lines"
)

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain the {topic} in detail in 200 words"
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

parser = StrOutputParser()

chains = RunnableParallel(
    short=RunnableLambda(lambda x: x["short"]) | short_template | model | parser,

    detailed=RunnableLambda(lambda x: x["detailed"]) | detailed_prompt | model | parser
)

result = chains.invoke({
    "short": {"topic": "Machine learning"},
    "detailed": {"topic": "Deep learning"}
})

print("SHORT:")
print(result["short"])

print("\nDETAILED:")
print(result["detailed"])