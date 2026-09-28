from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from rich import print


@tool
def get_text_len(text: str) -> int:
    """Get the length of the given text."""
    return len(text)


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

llm_with_tool = llm.bind_tools([get_text_len])

messages = [
    HumanMessage(content=input("You: "))
]

# 1. Ask the LLM
result = llm_with_tool.invoke(messages)

# Add AI response
messages.append(result)

# 2. Check whether LLM requested a tool
if result.tool_calls:

    for tool_call in result.tool_calls:

        if tool_call["name"] == "get_text_len":

            # Get arguments
            args = tool_call["args"]

            # Execute Python function
            output = get_text_len.invoke(args)

            # Send result back to LLM
            messages.append(
                ToolMessage(
                    content=str(output),
                    tool_call_id=tool_call["id"]
                )
            )

    # 3. Ask LLM to generate final answer
    final_result = llm_with_tool.invoke(messages)

    print(final_result.content)

else:
    # LLM answered without using a tool
    print(result.content)