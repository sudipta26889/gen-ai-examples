from langchain.agents import create_agent
from langchain.agents.middleware.types import wrap_tool_call
from langchain_core.messages import ToolMessage
from utils.llm import LMSTUDIO_LLM
from langchain.tools import tool
from ddgs import DDGS
from langgraph.config import get_stream_writer
from tools_for_agents import duckduckgo_search, calculate
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig


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

agent = create_agent(
    model=LMSTUDIO_LLM,
    tools=[duckduckgo_search, calculate],
    middleware=[
        handle_tool_errors,
        HumanInTheLoopMiddleware(
            interrupt_on={
                "duckduckgo_search": False,
                "calculate": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                }
            }
        )
    ],
    system_prompt="You are a helpful assistant. You always call search tool to find Latest temparature in celcius ONLY, of a given city input by user. And then do another search tool call to get the celcius to farenhight conversion formula. Then call the calculate tool to find the farenhight value. YOU should always use both search and calculate tool before answering, DO NOT ANSWER without both tool calls.",
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

# response = agent.invoke({"messages": "How is Kolkata today?"}, config)

# print(response["messages"])


def handle_human_in_loop():
    """Handle the human-in-the-loop workflow with proper interrupt handling."""
    
    # Start the agent stream
    stream = agent.stream(
        {"messages": [{"role": "user", "content": "How is Kolkata today?"}]},
        stream_mode="updates",
        config=config,
    )
    
    for chunk in stream:
        for step, data in chunk.items():
            print(f"\n=== STEP: {step} ===")
            
            if step == "__interrupt__":
                print("🛑 AGENT INTERRUPTED - Waiting for human approval")
                print("Available decisions: approve, edit, reject")
                
                # Get the pending tool call from the interrupt data
                if data and 'messages' in data:
                    last_message = data['messages'][-1]
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        tool_call = last_message.tool_calls[0]
                        print(f"Tool to approve: {tool_call['name']}")
                        print(f"Tool arguments: {tool_call['args']}")
                
                # Simulate human decision (in real app, this would be user input)
                decision = input("\nEnter your decision (approve/edit/reject): ").strip().lower()
                
                if decision == "approve":
                    print("✅ Tool call approved - continuing...")
                    # Continue with the approved tool call
                    stream = agent.stream(
                        None,  # No new input, just continue
                        stream_mode="updates", 
                        config=config
                    )
                elif decision == "edit":
                    print("✏️ Tool call needs editing")
                    # You could modify the tool call arguments here
                    new_args = input("Enter new arguments (JSON format): ")
                    # This would require more complex handling to modify the tool call
                    print("Continuing with original call...")
                    stream = agent.stream(None, stream_mode="updates", config=config)
                elif decision == "reject":
                    print("❌ Tool call rejected - stopping execution")
                    break
                else:
                    print("Invalid decision, defaulting to approve")
                    stream = agent.stream(None, stream_mode="updates", config=config)
                    
            else:
                # Handle normal streaming chunks
                if data is not None and 'messages' in data and data['messages']:
                    last_message = data['messages'][-1]
                    if hasattr(last_message, 'content_blocks'):
                        print(f"Content blocks: {last_message.content_blocks}")
                    elif hasattr(last_message, 'content'):
                        print(f"Content: {last_message.content}")
                    elif hasattr(last_message, 'tool_calls'):
                        print(f"Tool calls: {last_message.tool_calls}")
                else:
                    print(f"Data: {data}")

# Run the human-in-the-loop handler
handle_human_in_loop()
