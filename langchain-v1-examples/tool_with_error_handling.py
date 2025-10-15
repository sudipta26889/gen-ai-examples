from langchain.agents import create_agent
from langchain.agents.middleware.types import wrap_tool_call
from langchain_core.messages import ToolMessage
from utils.llm import LMSTUDIO_LLM
from langchain.tools import tool
from ddgs import DDGS
from langgraph.config import get_stream_writer  



@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

@tool
def search(query: str) -> str:
    """Search for information in the web."""
    writer = get_stream_writer()
    # stream any arbitrary data
    writer(f"Searching for: {query}")
    ddgs_results = DDGS().text(query, max_results=5)
    writer(f"Result From DuckDuckGo: {ddgs_results}")
    return ddgs_results

@tool
def calculate(x: str) -> str:
    """Useful for doing math calculations."""
    writer = get_stream_writer()
    writer(f"Calculating: {x}")
    result = eval(x)
    writer(f"Result: {result}")
    return result

agent = create_agent(
    model=LMSTUDIO_LLM,
    tools=[search, calculate],
    middleware=[handle_tool_errors],
    system_prompt="You are a helpful assistant. You always call search tool to find Latest temparature in celcius ONLY, of a given city input by user. And then do another search tool call to get the celcius to farenhight conversion formula. Then call the calculate tool to find the farenhight value. YOU should always use both search and calculate tool before answering, DO NOT ANSWER without both tool calls.",
)

# Run the agent
for chunk in agent.stream(  
    {"messages": [{"role": "user", "content": "How is Kolkata today?"}]},
    stream_mode="updates",
):
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data['messages'][-1].content_blocks}")
