from langchain.messages import AnyMessage
from langchain.agents import create_agent, AgentState
from langgraph.runtime import get_runtime
from typing import TypedDict
from utils.llm import LMSTUDIO_LLM


class CustomContext(TypedDict):
    user_name: str


from langchain.agents.middleware import dynamic_prompt, ModelRequest

def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is always sunny!"


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    return system_prompt


agent = create_agent(
    model=LMSTUDIO_LLM,
    tools=[get_weather],
    middleware=[dynamic_system_prompt],
    context_schema=CustomContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    context=CustomContext(user_name="Sudipta Dhara"),
)
for msg in result["messages"]:
    print(msg.pretty_print())