# core/services/agents/state.py

from typing import TypedDict


class AgentState(TypedDict):
    repo_id: int
    repo_url: str
    user_skills: list[dict]  # [{"skill":"python","band":"intermediate"}]
    intent: str  # "learn" or "vibe"
    selected_issue: dict | None
    conversation_phase: str  # onboarding/analysis/guidance/review
    messages: list[dict]  # full chat history
    recommendations: list[dict]  # filled by Issue Analysis Agent
    understanding_score: str  # SUFFICIENT / INSUFFICIENT
    user_approach: str | None
