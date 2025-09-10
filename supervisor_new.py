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


# --- Follow-up routing using rules + LLM fallback ---
def should_continue(state: MultiAgentState) -> str:
    if not state["messages"]:
        return "END"

    last = state["messages"][-1]
    content = getattr(last, "content", "")
    msg_type = type(last).__name__
    content_lower = content.lower()

    state["feedback"].append(f"[Supervisor] Last message: [{msg_type}] {content}")

    # --- Step 1: Rule-based routing ---
    if "no data" in content_lower or "not found" in content_lower:
        decision = "github_node"
    elif "github" in content_lower:
        decision = "knowledge_node"
    elif "record found" in content_lower or "✅" in content_lower:
        decision = "END"
    else:
        decision = None

    if decision:
        state["feedback"].append(f"[Supervisor] Rule-based route to: {decision}")
        return decision

    # --- Step 2: Gemini fallback ---
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
Reply ONLY with the node name. If the task is complete, reply END.
"""

    decision = VertexAI().getVertexModel().invoke(prompt).strip().lower()
    print("LLM decision (fallback):", decision)

    state["feedback"].append(f"[Supervisor] Gemini fallback route to: {decision}")
    return decision if decision in AGENT_REGISTRY else "END"


# --- Test Harness ---
if __name__ == "__main__":
    print("✅ Supervisor Test Start")

    class FakeHumanMessage:
        def __init__(self, content): self.content = content

    class FakeToolMessage:
        def __init__(self, content): self.content = content

    # --- Step 1: Simulate user query ---
    state: MultiAgentState = {
        "messages": [FakeHumanMessage("Check the database for record 123")],
        "feedback": [],
        "current_node": None,
        "next_node": None,
        "visited": []
    }

    print("\n🚦 Testing supervisor_router")
    next_node = supervisor_router(state)
    print("➡️  Next node:", next_node)
    print("📋 Feedback:", state["feedback"])

    # --- Step 2: Simulate DB node response ---
    print("\n🔁 Simulating DB output (no data)...")
    state["messages"].append(FakeToolMessage("No data found for record 123"))
    state["current_node"] = next_node
    state["visited"].append(next_node)

    print("\n🔁 Testing should_continue")
    next_node = should_continue(state)
    print("➡️  Next node:", next_node)
    print("📋 Feedback:", state["feedback"])

    print("\n✅ Supervisor Test Complete")
