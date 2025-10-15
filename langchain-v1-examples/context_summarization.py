from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

from utils.llm import LMSTUDIO_LLM

checkpointer = InMemorySaver()

agent = create_agent(
    model=LMSTUDIO_LLM,
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model=LMSTUDIO_LLM,
            max_tokens_before_summary=4000,  # Trigger summarization at 4000 tokens
            messages_to_keep=20,  # Keep last 20 messages after summary
        )
    ],
    checkpointer=checkpointer,
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": "hi, my name is Dhara"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

print(final_response["messages"][-1].pretty_print())
