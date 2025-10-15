from utils.llm import LMSTUDIO_LLM
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from tools_for_agents import web_search_using_serper

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model=LMSTUDIO_LLM,
    tools=[web_search_using_serper],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

print(result["structured_response"])
