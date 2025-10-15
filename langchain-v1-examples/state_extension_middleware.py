from utils.llm import LMSTUDIO_LLM
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from langchain.tools import InjectedState

class CustomAgentState(AgentState):  
    user_id: str
    preferences: dict

class StateExtensionMiddleware(AgentMiddleware[CustomAgentState]):
    state_schema = CustomAgentState  

def get_user_id(
    state: Annotated[CustomAgentState, InjectedState]
) -> str:
    """Look up user id."""
    user_id = state["user_id"]
    return user_id

def get_user_preferences(
    state: Annotated[CustomAgentState, InjectedState]
) -> str:
    """Look up user preferences."""
    preferences = state["preferences"]
    return preferences

agent = create_agent(
    LMSTUDIO_LLM,
    [get_user_id, get_user_preferences],
    middleware=[StateExtensionMiddleware()],  
    checkpointer=InMemorySaver(),
)

# Custom state can be passed in invoke
result = agent.invoke({
    "messages": [{"role": "user", "content": "Hello! Do you know my user id and preferences?"}],
    "user_id": "Sudipta",  
    "preferences": {"theme": "dark"}  
}, config={"configurable": {"thread_id": "1"}})

print(result)
