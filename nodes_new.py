from langgraph.graph import StateGraph, END
from src.state import MultiAgentState
from planner_executor import planner_node, executor_node
from src.nodes import database_node, github_node, knowledge_node

# -----------------------------
# Build the Graph
# -----------------------------
builder = StateGraph(MultiAgentState)

# Add planner and executor nodes
builder.add_node("planner_node", planner_node)
builder.add_node("executor_node", executor_node)

# Add actual worker nodes (they run the logic, based on state["next_node"])
builder.add_node("database_node", database_node)
builder.add_node("github_node", github_node)
builder.add_node("knowledge_node", knowledge_node)

# -----------------------------
# Set Entry Point
# -----------------------------
builder.set_entry_point("planner_node")

# -----------------------------
# Define Flow
# -----------------------------
# Planner → Executor
builder.add_edge("planner_node", "executor_node")

# Executor → next selected agent
builder.add_conditional_edges(
    "executor_node",
    lambda state: state.get("next_node", "END")
)

# Each agent returns to executor to continue the plan
builder.add_edge("database_node", "executor_node")
builder.add_edge("github_node", "executor_node")
builder.add_edge("knowledge_node", "executor_node")

# -----------------------------
# Compile Graph
# -----------------------------
graph = builder.compile()
