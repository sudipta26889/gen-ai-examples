from langchain.agents import create_agent
from tools_for_agents import web_search_using_serper
from utils.llm import LMSTUDIO_LLM

llm = LMSTUDIO_LLM

agent = create_agent(
    llm,
    tools=[web_search_using_serper],
    system_prompt="You are a helpful assistant",
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Kolkata"}]}
)

print(response)
