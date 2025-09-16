import os
import json
from typing import Optional
from typing_extensions import TypedDict, Annotated

from langgraph.schema import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.prebuilt import add_messages
from LLM.Gemini import VertexAI

# -----------------------------
# 🧠 Load Agent Registry
# -----------------------------
REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "config", "agent_registry.json")
with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
    AGENT_REGISTRY = json.load(f)

# -----------------------------
# 📦 Shared State Definition
# -----------------------------
class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    feedback: list[str]
    current_node: Optional[str]
    next_node: Optional[str]
    visited: list[str]

# -----------------------------
# 🔍 Get Last User Input
# -----------------------------
def get_last_user_input(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content.strip()
    return ""

# -----------------------------
# 🧾 Format Agent Outputs (Tool or AI)
# -----------------------------
def get_agent_outputs_grouped(state: MultiAgentState) -> dict[str, str]:
    outputs = {}
    visited = state.get("visited", [])

    tool_ai_messages = [msg for msg in state["messages"] if isinstance(msg, (ToolMessage, AIMessage))]

    for i, msg in enumerate(tool_ai_messages):
        if i < len(visited):
            node = visited[i]
            if node not in outputs:  # only first response per agent
                outputs[node] = msg.content.strip()
    return outputs

# -----------------------------
# 🤖 LLM-Based Supervisor Router
# -----------------------------
def dynamic_supervisor_router(state: MultiAgentState) -> str:
    user_input = get_last_user_input(state["messages"])
    visited = set(state.get("visited", []))

    # Determine unvisited agents
    available_agents = [name for name in AGENT_REGISTRY if name not in visited]
    if not available_agents:
        state["feedback"].append("All agents visited. Routing to END.")
        return "END"

    # Build agent descriptions
    agent_descriptions = "\n".join(
        f"- {name}: {AGENT_REGISTRY[name]['prompt']}"
        for name in available_agents
    )

    # Format agent outputs
    outputs_by_agent = get_agent_outputs_grouped(state)
    output_summary = "\n".join(
        f"- {agent}: {response}" for agent, response in outputs_by_agent.items()
    ) or "No agent responses yet."

    # Compose Gemini prompt
    prompt = f"""
You are the supervisor of a multi-agent system.

User originally asked:
{user_input}

Agent responses so far:
{output_summary}

Agents you can choose from:
{agent_descriptions}

Choose the most appropriate agent to continue, or reply END if the task is complete.
Reply with ONLY one of: {", ".join(available_agents + ["END"])}
""".strip()

    # Call Gemini
    decision = VertexAI().getVertexModel().invoke(prompt).strip().lower()
    print("🤖 LLM Decision:", decision)

    state["feedback"].append(f"Supervisor decided: {decision}")
    return decision if decision in available_agents + ["end"] else "END"
