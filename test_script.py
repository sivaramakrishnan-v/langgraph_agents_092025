import os
from langgraph.graph import StateGraph
from langgraph.schema import BaseMessage, HumanMessage, ToolMessage
from typing_extensions import TypedDict, Annotated
from typing import Optional
from LLM.Gemini import VertexAI  # Use your actual Gemini wrapper
from langgraph.prebuilt import add_messages

# --- Shared State ---
class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    feedback: list[str]
    current_node: Optional[str]
    next_node: Optional[str]
    visited: list[str]

# --- Simulated database agent ---
def database_node(state: MultiAgentState) -> MultiAgentState:
    response = "database_node: No data found for invoice 123"  # Simulated result
    tool_msg = ToolMessage(content=response, tool_call_id="db-1")
    state["messages"].append(tool_msg)
    state["visited"].append("database_node")
    state["current_node"] = "database_node"
    return state

# --- Simulated knowledge agent ---
def knowledge_node(state: MultiAgentState) -> MultiAgentState:
    response = "knowledge_node: No relevant info found. Task complete."  # Simulated result
    tool_msg = ToolMessage(content=response, tool_call_id="kn-1")
    state["messages"].append(tool_msg)
    state["visited"].append("knowledge_node")
    state["current_node"] = "knowledge_node"
    return state

# --- Extract last user message ---
def get_last_human_message(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

# --- Parse tool outputs ---
def get_tool_outputs(state: MultiAgentState) -> dict[str, str]:
    outputs = {}
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            if ":" in msg.content:
                node, content = msg.content.split(":", 1)
                outputs[node.strip()] = content.strip()
    return outputs

# --- Initial supervisor (force to database) ---
def supervisor_router(state: MultiAgentState) -> str:
    state["feedback"].append("Initial route: database_node")
    return "database_node"

# --- Follow-up supervisor logic ---
def should_continue(state: MultiAgentState) -> str:
    user_input = get_last_human_message(state["messages"]).strip()
    last_output = state["messages"][-1].content.strip().lower()

    outputs = get_tool_outputs(state)
    db_result = outputs.get("database_node", "").lower()
    kn_result = outputs.get("knowledge_node", "").lower()

    if "no data" in db_result or "no record" in db_result:
        state["feedback"].append("DB failed → try knowledge_node")
        return "knowledge_node"

    if "task complete" in kn_result or "no relevant" in kn_result:
        state["feedback"].append("Knowledge fallback triggered → END")
        return "END"

    if len(state["visited"]) >= 2:
        state["feedback"].append("Too many hops → END")
        return "END"

    # fallback to LLM
    prompt = f"""
User asked: {user_input}
Last response: {last_output}

Available agents: database_node, knowledge_node
Who should run next? Reply with ONLY one node or END.
"""
    decision = VertexAI().getVertexModel().invoke(prompt).strip().lower()
    print("🤖 LLM fallback decision:", decision)

    state["feedback"].append(f"Should continue to: {decision}")
    return decision if decision in ["database_node", "knowledge_node"] else "END"

# --- Build the graph ---
def build_graph():
    builder = StateGraph(MultiAgentState)

    builder.set_entry_point("supervisor")

    builder.add_node("supervisor", supervisor_router)
    builder.add_node("database_node", database_node)
    builder.add_node("knowledge_node", knowledge_node)

    builder.add_conditional_edges("supervisor", supervisor_router, {
        "database_node": "database_node",
        "knowledge_node": "knowledge_node"
    })

    builder.add_conditional_edges("database_node", should_continue, {
        "knowledge_node": "knowledge_node",
        "END": "__end__"
    })

    builder.add_conditional_edges("knowledge_node", should_continue, {
        "END": "__end__"
    })

    return builder.compile()

# --- Run test ---
if __name__ == "__main__":
    graph = build_graph()

    initial_state: MultiAgentState = {
        "messages": [HumanMessage(content="Can you check invoice 123 in the system?")],
        "feedback": [],
        "current_node": None,
        "next_node": None,
        "visited": [],
    }

    print("🔁 Starting graph stream...\n")
    for step in graph.stream(initial_state):
        print(f"📍 Step: {step['node']}")
        print(f"🧠 Feedback: {step['state']['feedback']}")
        print("🗨️  Messages:")
        for msg in step["state"]["messages"]:
            print(f"   - [{msg.__class__.__name__}] {msg.content}")
        print("-" * 40)
        if step["node"] == "__end__":
            print("✅ Reached END of graph.")
            break
