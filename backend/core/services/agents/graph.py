# backend/core/services/agents/graph.py
from langgraph.graph import END, StateGraph

from .nodes import (
    code_assist_node,
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
    messages = state.get("messages", [])
    if messages and messages[-1].get("role") == "assistant":
        return END
    return "onboarding"


def route_after_analysis(state: AgentState) -> str:
    phase = state.get("conversation_phase", "analysis")
    if phase == "guidance":
        return "guidance"
    if phase == "review":
        return "review"
    if phase in ("complete", "waiting"):
        return END
    messages = state.get("messages", [])
    if messages and messages[-1].get("role") == "assistant":
        return END
    return "issue_analysis"


def route_after_guidance(state: AgentState) -> str:
    phase = state.get("conversation_phase", "guidance")
    if phase == "code_assist":
        return "code_assist"
    if phase == "review":
        return "review"
    if phase == "complete":
        return END
    messages = state.get("messages", [])
    if messages and messages[-1].get("role") == "assistant":
        return END
    return "guidance"


def router_node(state: AgentState) -> AgentState:
    return state


def route_entry(state: AgentState) -> str:
    phase = state.get("conversation_phase", "onboarding")
    if phase == "complete":
        return END
    if phase == "review":
        return "review"
    if phase in ("guidance",) or state.get("selected_issue"):
        return "guidance"
    if state.get("user_skills"):
        return "issue_analysis"
    return "onboarding"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("onboarding", onboarding_node)
    graph.add_node("issue_analysis", issue_analysis_node)
    graph.add_node("guidance", guidance_node)
    graph.add_node("code_assist", code_assist_node)
    graph.add_node("review", review_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_entry,
        {
            "onboarding": "onboarding",
            "issue_analysis": "issue_analysis",
            "guidance": "guidance",
            "review": "review",
            END: END,
        },
    )

    graph.add_conditional_edges(
        "onboarding",
        route_after_onboarding,
        {
            "onboarding": "onboarding",
            "issue_analysis": "issue_analysis",
            END: END,
        },
    )

    graph.add_conditional_edges(
        "issue_analysis",
        route_after_analysis,
        {
            "issue_analysis": "issue_analysis",
            "guidance": "guidance",
            "review": "review",
            END: END,
        },
    )

    graph.add_conditional_edges(
        "guidance",
        route_after_guidance,
        {
            "guidance": "guidance",
            "code_assist": "code_assist",
            "review": "review",
            END: END,
        },
    )

    graph.add_conditional_edges(
        "code_assist",
        route_after_guidance,
        {
            "guidance": "guidance",
            "code_assist": "code_assist",
            "review": "review",
            END: END,
        },
    )
    graph.add_edge("review", END)

    return graph.compile()
