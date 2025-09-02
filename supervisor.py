# src/supervisor.py

import os
import json
from typing import Optional, Dict
from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import add_messages
from langchain_core.messages import BaseMessage, HumanMessage

from nodes import AGENT_NODES             # name -> node_fn (defined in nodes.py)
from LLM.Gemini import VertexAI           # your Vertex wrapper


# =========================
# 1) Shared state schema
# =========================
class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    feedback: list[str]
    current_node: Optional[str]
    next_node: Optional[str]
    visited: list[str]


# =========================
# 2) Registry loader
# =========================
REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "config", "agent_registry.json")

def load_agent_registry(path: str) -> Dict[str, dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Agent registry file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # soft validation (schema-lite): require `prompt` and recommend `agent`
    missing = [
        name for name, info in data.items()
        if not isinstance(info, dict) or "prompt" not in info
    ]
    if missing:
        raise ValueError(
            "Invalid agent_registry.json. Missing `prompt` field for: "
            + ", ".join(missing)
        )
    return data

AGENT_REGISTRY = load_agent_registry(REGISTRY_PATH)


# =========================
# 3) LLM-based supervisor
# =========================
def supervisor(state: MultiAgentState) -> str:
    """
    Router:
      - On first hop: routes using the user's message.
      - On later hops: routes using the last node's output (the last message in state).
      - Writes a simple trace to state['feedback'].
    """

    # Get the latest message (Human/AI/Tool)
    if state["messages"]:
        latest_msg = state["messages"][-1]
    else:
        latest_msg = None

    if latest_msg:
        last_text = getattr(latest_msg, "content", "")
    else:
        last_text = ""

    # Build agent descriptions from registry
    agent_descriptions = "\n".join(
        f"- {node_name}: {info['prompt']}"
        for node_name, info in AGENT_REGISTRY.items()
    )

    routing_prompt = f"""
You are the supervisor of a multi-agent system.

Available agents:
{agent_descriptions}

The last message was:
\"\"\"{last_text}\"\"\"

Which agent should handle this next?
Reply with ONLY the agent node name exactly as listed above (e.g., database_node). Do not add extra words.
"""

    model = VertexAI().getVertexModel()
    decision = model.invoke(routing_prompt).strip().lower()

    if decision not in AGENT_REGISTRY:
        decision = "END"

    # trace for debugging/audit
    state["feedback"].append(f"Supervisor routed to: {decision}")
    return decision


# =========================
# 4) Graph builder
# =========================
def build_supervisor_graph():
    """
    Build a graph with:
      - One node per entry in agent_registry.json
      - Each node returns to the 'supervisor' router
      - The router decides the next node name via LLM
    """
    builder = StateGraph(MultiAgentState)

    # Add nodes dynamically, ensuring node function exists
    for node_name in AGENT_REGISTRY.keys():
        node_fn = AGENT_NODES.get(node_name)
        if node_fn is None:
            raise ValueError(
                f"Missing node function for '{node_name}'. "
                f"Define it in nodes.py and expose it via AGENT_NODES."
            )
        builder.add_node(node_name, node_fn)
        builder.add_edge(node_name, "supervisor")

    # Router mapping
    routing_map = {name: name for name in AGENT_REGISTRY.keys()}
    routing_map["END"] = END

    builder.add_conditional_edges("supervisor", supervisor, routing_map)
    builder.set_entry_point("supervisor")

    return builder.compile()
