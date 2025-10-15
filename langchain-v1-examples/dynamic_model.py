from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest
from langchain.agents.middleware.types import ModelResponse

basic_model = ChatOpenAI(
    model="openai/gpt-oss-20b",
    temperature=0.1,
    api_key="no-key-needed-for-lm-studio",
    base_url="http://127.0.0.1:1234/v1",
    max_tokens=120000,
    streaming=True,
    verbose=True,
)
advanced_model = ChatOpenAI(
    model="openai/gpt-oss-120b",
    temperature=0.1,
    api_key="no-key-needed-for-lm-studio",
    base_url="http://192.168.11.108:1234/v1",
    max_tokens=120000,
    streaming=True,
    verbose=True,
)

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    request.model = model
    return handler(request)

agent = create_agent(
    model=basic_model,  # Default model
    middleware=[dynamic_model_selection]
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the capital of paris, germany, india, USA, london"}]}
)

print(response)
