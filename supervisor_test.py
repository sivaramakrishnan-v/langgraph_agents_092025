# supervisor.py
import os
import json
from typing import Optional
from typing_extensions import TypedDict, Annotated

from langgraph.schema import BaseMessage, HumanMessage
from langgraph.prebuilt import add_messages
from LLM.Gemini import VertexAI

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "config", "agent_registry.json")
with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
    AGENT_REGISTRY = json.load(f)

class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    feedback: list[str]
    current_node: Optional[str]
    next_node: Optional[str]
    visited: list[str]

def supervisor_router(state: MultiAgentState) -> str:
    """Initial router for user input."""
    user_input = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break

    descriptions = "\n".join(
        f"- {k}: {v['prompt']}" for k, v in AGENT_REGISTRY.items()
    )

    prompt = f"""
You are the supervisor of a multi-agent system.
Available agents:
{descriptions}

The user asked:
{user_input}

Which agent should handle this?
Reply with ONLY the node name.
"""
    decision = VertexAI().getVertexModel().invoke(prompt).strip().lower()
    state["feedback"].append(f"Initial route: {decision}")
    return decision if decision in AGENT_REGISTRY else "END"

def should_continue(state: MultiAgentState) -> str:
    """Handles routing after each node's output."""
    if not state["messages"]:
        return "END"

    last = state["messages"][-1]
    content = getattr(last, "content", "")
    msg_type = type(last).__name__

    descriptions = "\n".join(
        f"- {k}: {v['prompt']}" for k, v in AGENT_REGISTRY.items()
    )

    prompt = f"""
You are the supervisor of a multi-agent system.

Available agents:
{descriptions}

The last message was from: {msg_type}
{content}

Which agent should handle this next?
Reply with ONLY the node name. If task is done, reply END.
"""
    decision = VertexAI().getVertexModel().invoke(prompt).strip().lower()
    state["feedback"].append(f"Should continue to: {decision}")
    return decision if decision in AGENT_REGISTRY else "END"

if __name__ == "__main__":
    # Dummy message stubs for test
    class FakeHumanMessage:
        def __init__(self, content):
            self.content = content

    class FakeToolMessage:
        def __init__(self, content):
            self.content = content

    # Simulated initial state
    state: MultiAgentState = {
        "messages": [FakeHumanMessage("Check the database for record 123")],
        "feedback": [],
        "current_node": None,
        "next_node": None,
        "visited": []
    }

    print("\n🚦 Testing supervisor_router")
    next_node = supervisor_router(state)
    print("Next node:", next_node)
    print("Feedback:", state["feedback"])

    # Simulate a node response
    state["messages"].append(FakeToolMessage("Found record 123 in database"))
    state["current_node"] = next_node
    state["visited"].append(next_node)

    print("\n🔁 Testing should_continue")
    next_node = should_continue(state)
    print("Next node:", next_node)
    print("Feedback:", state["feedback"])
