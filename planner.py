import os
import json
from typing import Optional, Dict, Any
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
    plan: Optional[list[str]]
    current_step: Optional[int]


# -----------------------------
# 🔍 Last user query
# -----------------------------
def get_last_user_input(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content.strip()
    return ""


# -----------------------------
# 🧾 Output summary for prompt
# -----------------------------
def get_agent_outputs_grouped(state: MultiAgentState) -> dict[str, str]:
    outputs = {}
    visited = state.get("visited", [])

    tool_ai_messages = [msg for msg in state["messages"] if isinstance(msg, (ToolMessage, AIMessage))]

    for i, msg in enumerate(tool_ai_messages):
        if i < len(visited):
            node = visited[i]
            content = msg.content.strip()
            if content and node not in outputs:
                outputs[node] = content
    return outputs


# -----------------------------
# 🧠 Planner Node (LLM makes a full plan)
# -----------------------------
def planner_node(state: MultiAgentState) -> MultiAgentState:
    user_input = get_last_user_input(state["messages"])
    visited = set(state.get("visited", []))

    available_agents = [name for name in AGENT_REGISTRY if name not in visited]
    if not available_agents:
        state["feedback"].append("All agents visited. Routing to END.")
        state["plan"] = []
        state["current_step"] = 0
        state["next_node"] = "END"
        return state

    agent_descriptions = "\n".join(
        f"- {name}: {AGENT_REGISTRY[name]['prompt']}"
        for name in available_agents
    )

    outputs_by_agent = get_agent_outputs_grouped(state)
    output_summary = "\n".join(
        f"- {agent}: {response}" for agent, response in outputs_by_agent.items()
    ) or "No agent responses yet."

    prompt = f"""
You are the planner in a multi-agent system.

User originally asked:
{user_input}

Agent responses so far:
{output_summary}

Agents you can choose from:
{agent_descriptions}

Based on the above, generate a step-by-step plan of agent names to call in sequence.
Reply ONLY with a comma-separated list of node names. For example:
database_node, github_node
""".strip()

    plan_text = VertexAI().getVertexModel().invoke(prompt).strip().lower()
    print("🧠 Planner LLM returned:", plan_text)

    # Parse plan and filter valid steps
    plan = [step.strip() for step in plan_text.split(",") if step.strip() in AGENT_REGISTRY]
    state["plan"] = plan
    state["current_step"] = 0

    if not plan:
        state["feedback"].append("Planner could not generate a valid plan. Routing to END.")
        state["next_node"] = "END"
    else:
        state["feedback"].append(f"Planner created plan: {plan}")

    return state


# -----------------------------
# 🧭 Executor Node (routes to next)
# -----------------------------
def executor_node(state: MultiAgentState) -> MultiAgentState:
    plan = state.get("plan", [])
    step = state.get("current_step", 0)

    if step >= len(plan):
        print("✅ Plan completed.")
        state["next_node"] = "END"
        return state

    next_node = plan[step]
    state["next_node"] = next_node
    state["current_node"] = next_node
    state["visited"].append(next_node)
    state["current_step"] += 1
    print(f"🚀 Executor routing to: {next_node}")
    return state
