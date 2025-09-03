# nodes.py

from langchain_core.messages import ToolMessage
from src.agents import get_db_agent
from supervisor import MultiAgentState


def database_node(state: MultiAgentState) -> MultiAgentState:
    agent = get_db_agent()
    result = agent.invoke(state)  # Expects {"messages": [...]}

    # Safely extract first ToolMessage from result
    tool_messages = [
        msg for msg in result["messages"] if isinstance(msg, ToolMessage)
    ]

    if tool_messages:
        tool_msg = tool_messages[0]
        feedback_msg = f"[DB Agent] Tool responded: {tool_msg.content}"
    else:
        feedback_msg = "[DB Agent] No ToolMessage found in response."

    # Return updated state
    return {
        "messages": result["messages"],
        "feedback": state["feedback"] + [feedback_msg],
        "current_node": "database_node",
        "next_node": None,
        "visited": state.get("visited", []) + ["database_node"]
    }
