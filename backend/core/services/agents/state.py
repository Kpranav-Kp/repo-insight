# core/services/agents/state.py
from typing import NotRequired, TypedDict


class AgentState(TypedDict):
    repo_id: int
    repo_url: str
    user_skills: list[dict]  # [{"skill": "python", "band": "intermediate"}]
    selected_issue: dict | None
    code_assist_count: int
    stuck_counter: int
    conversation_phase: str  # onboarding | analysis | guidance | review | learning
    messages: list[
        dict
    ]  # full chat history  {"role": "user"|"assistant", "content": "..."}
    recommendations: list[dict]  # filled by Issue Analysis Agent
    understanding_score: str  # SUFFICIENT | INSUFFICIENT
    user_approach: str | None
    weak_skills: NotRequired[list[str]]  # skills at heard_of level for mixed users
    user_id: int
    session_id: int


def is_beginner(user_skills: list[dict]) -> bool:
    """True if all skills are heard_of or user_skills is empty."""
    if not user_skills:
        return True
    return all(
        isinstance(s, dict) and s.get("band", "") == "heard_of" for s in user_skills
    )
