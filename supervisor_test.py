# supervisor.py
import os
import json
from typing import Optional
from typing_extensions import TypedDict, Annotated

from langgraph.schema import BaseMessage, HumanMessage
from langgraph.prebuilt import add_messages
from LLM.Gemini import VertexAI

# --- Load agent registry ---
REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "config", "agent_registry.json")
with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
    AGENT_REGISTRY = json.load(f)

# --- Define MultiAgentState ---
class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    feedback: list[str]
    current_node: Optional[str]
    next_node: Optional[str]
    visited: list[str]


# --- Initial router: force to database_node ---
def supervisor_router(state: MultiAgentState) -> str:
    state["feedback"].append("Initial route forced to: database_node")
    return "database_node"


# --- Follow-up routing using Gemini ---
def should_continue(state: MultiAgentState) -> str:
    if not state["messages"]:
        return "END"

    last = state["messages"][-1]
    content = getattr(last, "content", "")
    msg_type = type(last).__name__

    # ✅ Clear and readable descriptions
    descriptions_list = []
    for node_name, config in AGENT_REGISTRY.items():
        agent_purpose = config.get("prompt", "[No description]")
        descriptions_list.append(f"- {node_name}: {agent_purpose}")
    descriptions = "\n".join(descriptions_list)

    prompt = f"""
You are the supervisor of a multi-agent system.

Available agents:
{descriptions}

The last message was from: {msg_type}
{content}

Which agent should handle this next?
Reply with ONLY the node name. If the task is complete, reply END.
"""

    decision = VertexAI().getVertexModel().invoke(prompt).strip().lower()
    print("LLM decision (follow-up):", decision)

    state["feedback"].append(f"Should continue to: {decision}")
    return decision if decision in AGENT_REGISTRY else "END"


# --- Test harness ---
if __name__ == "__main__":
    class FakeHumanMessage:
        def __init__(self, content): self.content = content

    class FakeToolMessage:
        def __init__(self, content): self.content = content

    # Initial test state
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

    # Simulate DB response
    state["messages"].append(FakeToolMessage("No records found in database"))
    state["current_node"] = next_node
    state["visited"].append(next_node)

    print("\n🔁 Testing should_continue")
    next_node = should_continue(state)
    print("Next node:", next_node)
    print("Feedback:", state["feedback"])
