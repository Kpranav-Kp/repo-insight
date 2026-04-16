# core/services/agents/state.py
from typing import Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    repo_id: str
    repo_url: str
    user_skills: List[Dict]          # [{"skill": "python", "band": "intermediate"}]
    intent: str                       # "learn" | "vibe"
    selected_issue: Optional[Dict]
    conversation_phase: str           # onboarding | analysis | guidance | review
    messages: List[Dict]              # full chat history  {"role": "user"|"assistant", "content": "..."}
    recommendations: List[Dict]       # filled by Issue Analysis Agent
    understanding_score: str          # SUFFICIENT | INSUFFICIENT
    user_approach: Optional[str]
