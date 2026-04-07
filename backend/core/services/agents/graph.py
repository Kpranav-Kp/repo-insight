# backend/core/services/agents/graph.py

from langgraph.graph import END, StateGraph

from .nodes import (
    guidance_node,
    issue_analysis_node,
    onboarding_node,
    review_node,
)
from .state import AgentState


def route_after_onboarding(state: AgentState) -> str:
    phase = state.get("conversation_phase", "onboarding")
    if phase == "analysis":
        return "issue_analysis"
    return "onboarding"


def route_after_analysis(state: AgentState) -> str:
    phase = state.get("conversation_phase", "analysis")
    if phase == "guidance":
        return "guidance"
    if phase == "review":
        return "review"
    return "issue_analysis"


def route_after_guidance(state: AgentState) -> str:
    phase = state.get("conversation_phase", "guidance")
    if phase == "review":
        return "review"
    return "guidance"


def build_graph():
    graph = StateGraph(AgentState)

    # add all 4 agent nodes
    graph.add_node("onboarding", onboarding_node)
    graph.add_node("issue_analysis", issue_analysis_node)
    graph.add_node("guidance", guidance_node)
    graph.add_node("review", review_node)

    # entry point
    graph.set_entry_point("onboarding")

    # transitions
    graph.add_conditional_edges(
        "onboarding",
        route_after_onboarding,
        {
            "onboarding": "onboarding",
            "issue_analysis": "issue_analysis",
        },
    )

    graph.add_conditional_edges(
        "issue_analysis",
        route_after_analysis,
        {
            "issue_analysis": "issue_analysis",
            "guidance": "guidance",
            "review": "review",
        },
    )

    graph.add_conditional_edges(
        "guidance",
        route_after_guidance,
        {
            "guidance": "guidance",
            "review": "review",
        },
    )

    graph.add_edge("review", END)

    return graph.compile()
