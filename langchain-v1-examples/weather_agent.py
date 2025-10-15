from langchain.agents import create_agent
from tools_for_agents import web_search_using_serper
from utils.llm import LMSTUDIO_LLM
from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver


llm = LMSTUDIO_LLM

system_prompt = """
You are a helpful assistant
"""

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    weather_condition: str
    temperature: str
    hummidity: str
    wind_speed: str
    city: str
    response: str


checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    system_prompt=system_prompt,
    tools=[web_search_using_serper],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

# Let's say user_id is 1 and thread_id is 1

user_id = "abc"
thread_id = "1"


# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": thread_id}}


# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Kolkata?"}]},
    config=config,
    context=Context(user_id=user_id),
)

print(response['structured_response'])


# we can continue the conversation using the same `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id=user_id)
)

print(response['structured_response'])
