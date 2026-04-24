# backend/core/services/agents/nodes.py
"""
Four LangGraph agent nodes.

Onboarding  → collects repo_url, user_skills, intent
Analysis    → recommends issues; waits for user to pick one
Guidance    → Socratic loop; never gives code; exits when understanding = SUFFICIENT
Review      → validates approach, checks novelty, produces PR outline
"""

import json
import logging

from django.conf import settings

from ..graph_loader import load_engine_for_repo
from .state import AgentState
from .tools import fetch_contributing_guidelines, fetch_repo_skills

logger = logging.getLogger(__name__)


# ── LLM ───────────────────────────────────────────────────────────────────────


def get_llm():
    from langchain_groq import ChatGroq

    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=settings.GROQ_API_KEY,
        temperature=0.7,
    )


def llm_respond(system_prompt: str, messages: list[dict]) -> str:
    """Call the LLM and return plain text."""
    llm = get_llm()
    try:
        response = llm.invoke([{"role": "system", "content": system_prompt}, *messages])
        return str(response.content).strip()
    except Exception as exc:
        logger.exception("LLM call failed: %s", exc)
        return "I'm having trouble connecting right now. Please try again."


# ── HELPERS ───────────────────────────────────────────────────────────────────


def _last_user_message(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _assistant(messages: list[dict], text: str) -> list[dict]:
    """Append an assistant message and return the updated list."""
    return [*messages, {"role": "assistant", "content": text}]


def _repo_id_from_url(repo_url: str) -> str:
    """Derive a stable string key from a GitHub URL (owner/repo)."""
    parts = repo_url.rstrip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else repo_url


# ── AGENT 1: ONBOARDING ───────────────────────────────────────────────────────


def onboarding_node(state: AgentState) -> AgentState:
    """
    Collects user_skills and intent only.
    repo_url and repo_id are already in state from the session.
    """
    messages = state.get("messages") or []
    repo_url = state.get("repo_url")

    if not repo_url:
        reply = "Please provide the GitHub repository URL you want to contribute to."
        return {
            **state,
            "messages": _assistant(messages, reply),
            "conversation_phase": "onboarding",
        }

    # ── 2. Collect user_skills (no URL detection) ────────────────────────────
    if not state.get("user_skills"):
        if len(messages) <= 1:
            repo_skills = fetch_repo_skills.invoke(repo_url)
            reply = llm_respond(
                f"""
                The repository uses these skills most: {repo_skills}
                Ask the user naturally about their experience with these specific skills.
                Do NOT ask them to rate themselves on a numeric scale.
                Example: 'I see this repo uses Python and REST APIs a lot — how comfortable are you with those?'
                ONE question only.
                """,
                messages,
            )
            return {
                **state,
                "messages": _assistant(messages, reply),
                "conversation_phase": "onboarding",
            }

        parsed_raw = llm_respond(
            """
            Based on this conversation, extract the user's skills and experience level.
            Return ONLY valid JSON — a list of objects like:
            [{"skill": "python", "band": "intermediate"}, {"skill": "sql", "band": "beginner"}]

            Valid bands: beginner, intermediate, comfortable.
            Return an empty list [] if you cannot determine any skills.
            No markdown fences, no explanation — pure JSON only.
            """,
            messages,
        )
        try:
            clean = parsed_raw.replace("```json", "").replace("```", "").strip()
            skills = json.loads(clean)
            if not isinstance(skills, list):
                skills = []
        except (json.JSONDecodeError, ValueError):
            skills = []

        if skills:
            state = {**state, "user_skills": skills}
        else:
            reply = llm_respond(
                """
                The user's skills are still unclear. Ask a gentle follow-up to understand
                their programming background and experience. ONE question only.
                """,
                messages,
            )
            return {
                **state,
                "messages": _assistant(messages, reply),
                "conversation_phase": "onboarding",
            }

        reply = "Great! Let me find the best issues for you — one moment. 🔍"
        return {
            **state,
            "messages": _assistant(messages, reply),
            "conversation_phase": "analysis",
        }

    return {**state, "conversation_phase": "analysis"}


# ── AGENT 2: ISSUE ANALYSIS ───────────────────────────────────────────────────


def issue_analysis_node(state: AgentState) -> AgentState:
    """
    1. Loads the knowledge graph for the repo and recommends issues.
    2. Presents up to 3 issues in plain language.
    3. Waits for the user to pick one.
    """
    messages = state.get("messages") or []
    repo_id = state.get("repo_id")
    skills: list[dict] = state.get("user_skills") or []
    skill_names = [s["skill"] for s in skills]

    # ── If the user already picked an issue, route forward ───────────────────
    if state.get("selected_issue"):
        return {**state, "conversation_phase": "guidance"}

    # ── Build recommendations once ────────────────────────────────────────────
    if not state.get("recommendations"):
        try:
            engine = load_engine_for_repo(repo_id)
            raw_results = engine.recommend(skill_names, top_k=5)
        except Exception as exc:
            logger.warning("Engine not ready: %s", exc)
            reply = "The repository is still being analyzed. Please wait a moment and try again."
            return {
                **state,
                "messages": _assistant(messages, reply),
                "conversation_phase": "analysis",
            }
        # Filter by experience band
        band = skills[0].get("band", "beginner") if skills else "beginner"
        if band == "beginner":
            raw_results = [r for r in raw_results if len(r.get("skills", [])) <= 4]

        # Annotate with novelty score
        flagged = []
        for r in raw_results:
            try:
                score = engine.graph.novelty_score("", r["id"])
                r["novelty"] = round(score, 2)
                r["already_tried"] = score < 0.5
            except Exception:
                r["novelty"] = 1.0
                r["already_tried"] = False
            flagged.append(r)

        state = {**state, "recommendations": flagged}

    recommendations = state.get("recommendations") or []

    # ── Check if the user just picked an issue ────────────────────────────────
    last = _last_user_message(messages)
    if last and recommendations:
        picked = (
            llm_respond(
                f"""
            The user said: "{last}"
            Available issues (JSON): {json.dumps(recommendations)}

            Did the user clearly pick one of these issues?
            If YES, reply with ONLY the issue's id number (integer).
            If NO or ambiguous, reply with: none
            """,
                messages,
            )
            .strip()
            .lower()
        )

        if picked != "none":
            selected = next(
                (r for r in recommendations if str(r.get("id")) == picked), None
            )
            if selected:
                next_phase = "guidance"
                return {
                    **state,
                    "messages": messages,
                    "selected_issue": selected,
                    "conversation_phase": next_phase,
                }

    # ── Present issues ────────────────────────────────────────────────────────
    reply = llm_respond(
        f"""
        You are helping a developer pick a GitHub issue to work on.
        Here are the top matching issues (JSON): {json.dumps(recommendations[:3])}

        For each issue explain in plain, simple language:
          - What the issue is asking for
          - What skills are involved
          - Rough complexity (simple / moderate / involved)
          - If 'already_tried' is true, add a note: 'Note: a similar approach was attempted before.'

        Present at most 3 issues. Number them clearly.
        End by asking which one they would like to work on.
        Do NOT write any code.
        """,
        messages,
    )

    return {
        **state,
        "messages": _assistant(messages, reply),
        "conversation_phase": "analysis",
    }


# ── AGENT 3: GUIDANCE ─────────────────────────────────────────────────────────


def guidance_node(state: AgentState) -> AgentState:
    """
    Socratic loop — never gives code.

    Flow:
      1. Check for skill gaps → provide learning path if gaps exist.
      2. Ask a targeted understanding question.
      3. Evaluate the user's answer.
         - INSUFFICIENT → increment stuck_counter; if >=2 or user asks "stuck/hint", go to code_assist.
         - SUFFICIENT + genuine → advance to review.
         - SUFFICIENT + vibe-coded → ask a more specific follow-up.
    """
    messages = state.get("messages") or []
    selected_issue = state.get("selected_issue") or {}
    repo_url = state.get("repo_url", "")
    user_skills_list = [s["skill"] for s in (state.get("user_skills") or [])]
    issue_skills = selected_issue.get("skills") or []

    last = _last_user_message(messages)

    # ── Evaluate the user's latest answer ────────────────────────────────────
    if last:
        evaluation = llm_respond(
            f"""
            Issue context (JSON): {json.dumps(selected_issue)}
            User's answer: "{last}"

            Does this answer show SPECIFIC understanding of the issue?

            SUFFICIENT means:
              - References the actual problem described in the issue
              - Shows understanding of why the bug/problem happens
              - Suggests a general direction (not code)

            INSUFFICIENT means:
              - Vague or generic — could apply to any issue
              - No reference to the specific codebase or file
              - Reads like AI-generated boilerplate

            Reply with ONLY: SUFFICIENT  OR  INSUFFICIENT
            """,
            messages,
        ).upper()

        state = {**state, "understanding_score": evaluation}

        if evaluation == "SUFFICIENT":
            vibe_check = llm_respond(
                f"""
                User's explanation: "{last}"

                Does this read like:
                A) A genuine developer response with specific, concrete details
                B) Generic AI-generated text with no real specifics

                Reply ONLY: genuine  OR  vibe_coded
                """,
                messages,
            ).lower()

            if "genuine" in vibe_check:
                reply = (
                    "Great understanding! 🎉 Let me check your approach against "
                    "the repo's contributing guidelines."
                )
                return {
                    **state,
                    "messages": _assistant(messages, reply),
                    "conversation_phase": "review",
                }
            else:
                # Vibe-coded — demand specifics
                reply = llm_respond(
                    f"""
                    The user's answer seems too generic or AI-generated.
                    Issue: {json.dumps(selected_issue)}

                    Ask one very specific follow-up question that requires them to reference
                    an actual file name, function, or line of behaviour from the codebase.
                    Do NOT provide any code or hints.
                    """,
                    messages,
                )
                return {
                    **state,
                    "messages": _assistant(messages, reply),
                    "conversation_phase": "guidance",
                }

        # INSUFFICIENT — increment stuck counter and decide whether to offer code assist
        stuck = state.get("stuck_counter", 0) + 1
        # if user explicitly asks for help, or has been stuck twice, go to code assist
        if stuck >= 2 or "stuck" in last.lower() or "hint" in last.lower():
            return {
                **state,
                "stuck_counter": stuck,
                "conversation_phase": "code_assist",
            }
        else:
            reply = llm_respond(
                f"""
                Issue: {json.dumps(selected_issue)}
                User's answer: "{last}"

                Their answer was too vague. Ask ONE targeted follow-up question that pushes
                them to think about:
                  - The specific file or module involved
                  - The root cause of the problem
                Do NOT give any code or reveal the answer.
                """,
                messages,
            )
            return {
                **state,
                "stuck_counter": stuck,
                "messages": _assistant(messages, reply),
                "conversation_phase": "guidance",
            }

    # ── Skill-gap learning path (first visit, no answer yet) ─────────────────
    gaps = [s for s in issue_skills if s not in user_skills_list]
    if gaps:
        learning_path = llm_respond(
            f"""
            Developer knows: {user_skills_list}
            Issue requires : {issue_skills}
            Skill gaps     : {gaps}

            Create a concise learning path for each gap:
              - One concept to understand
              - One real resource URL (if you know a reliable one)
              - One small hands-on exercise
              - Estimated time

            No code. No solutions. End with: 'Come back after going through these!'
            """,
            messages,
        )
        return {
            **state,
            "messages": _assistant(messages, learning_path),
            "conversation_phase": "guidance",
        }

    # ── First understanding question ──────────────────────────────────────────
    # Fetch guidelines for context (used by LLM prompt, not returned to user directly)
    guidelines_context = ""
    try:
        guidelines_context = fetch_contributing_guidelines.invoke(repo_url)
    except Exception as e:
        logger.warning("Could not fetch guidelines: %s", e)
        guidelines_context = ""

    reply = llm_respond(
        f"""
        Issue (JSON): {json.dumps(selected_issue)}
        Repository  : {repo_url}
        Guidelines  : {guidelines_context[:500] if guidelines_context else "N/A"}

        Ask the user ONE specific question to check their understanding of this issue.
        The question should make them think about:
          - What is actually causing the problem
          - Where in the codebase the fix would live

        Do NOT give any code.
        Do NOT hint at the answer.
        Point them to WHERE to look, not WHAT to write.
        """,
        messages,
    )
    return {
        **state,
        "messages": _assistant(messages, reply),
        "conversation_phase": "guidance",
    }


# ── AGENT 4: REVIEW ───────────────────────────────────────────────────────────


def review_node(state: AgentState) -> AgentState:
    """
    Validates the user's approach before they write code.

    Produces:
      1. Approach Review   — what looks good / what might cause rejection
      2. Novelty Check     — warns if a similar approach was tried before
      3. PR Outline        — correct format for THIS repo, blanks for user to fill
    """
    messages = state.get("messages") or []
    selected_issue = state.get("selected_issue") or {}
    repo_url = state.get("repo_url", "")
    repo_id = state.get("repo_id")
    user_approach = state.get("user_approach") or ""

    # ── Fetch contributing guidelines ─────────────────────────────────────────
    guidelines = "No CONTRIBUTING.md found."
    try:
        guidelines = fetch_contributing_guidelines.invoke(repo_url)
    except Exception as exc:
        logger.warning("Could not fetch guidelines: %s", exc)

    # ── Novelty score ─────────────────────────────────────────────────────────
    novelty = 1.0
    if user_approach and repo_id:
        try:
            engine = load_engine_for_repo(repo_id)
            novelty = engine.graph.novelty_score(
                user_approach, selected_issue.get("id", "")
            )
        except Exception as exc:
            logger.warning("Novelty score failed: %s", exc)

    # ── Generate final review + PR outline ────────────────────────────────────
    reply = llm_respond(
        f"""
        Issue          : {json.dumps(selected_issue)}
        User's approach: {user_approach or "not yet described — ask them to describe it briefly"}
        Guidelines     : {guidelines}
        Novelty score  : {novelty:.2f}  (below 0.5 = similar approach tried before)

        Produce a structured output with these three sections:

        ## 1. APPROACH REVIEW
           - What the user is doing right
           - What might get the PR rejected
           - What to clarify with maintainers before starting

        ## 2. NOVELTY CHECK
           - If novelty < 0.5: warn clearly that a similar approach was tried;
             summarise what was attempted so the user can differentiate theirs.
           - If novelty ≥ 0.5: confirm the approach looks original.

        ## 3. PR OUTLINE TEMPLATE
           - Use the correct format for this specific repo (based on guidelines)
           - Include: issue reference, change summary, testing notes, checklist
           - Leave blanks with [FILL IN] markers for the user to complete
           - Do NOT write actual code changes

        Guide them — do not write code for them.
        """,
        messages,
    )

    return {
        **state,
        "messages": _assistant(messages, reply),
        "conversation_phase": "complete",
    }


def code_assist_node(state: AgentState) -> AgentState:
    """
    Provides boilerplate with TODOs, limited to MAX_ASSISTS per session.
    Increments code_assist_count; refuses after limit.
    """
    messages = state.get("messages") or []
    selected_issue = state.get("selected_issue") or {}
    repo_url = state.get("repo_url", "")
    code_assist_count = state.get("code_assist_count", 0)
    MAX_ASSISTS = 3

    if code_assist_count >= MAX_ASSISTS:
        reply = (
            "You've already received code assistance a few times. "
            "Open source contribution is about learning by doing. "
            "Try to complete the TODOs from the previous snippet or ask me a specific question. "
            "I won't provide more code to ensure you truly understand the process."
        )
        return {
            **state,
            "messages": _assistant(messages, reply),
            "conversation_phase": "guidance",
            "code_assist_count": code_assist_count,
        }

    # Generate boilerplate with TODOs
    reply = llm_respond(
        f"""
        Issue: {json.dumps(selected_issue)}
        Repository: {repo_url}
        User's previous understanding: {state.get("understanding_score", "unknown")}

        The user is stuck. Provide a **boilerplate code snippet** that outlines the structure needed.
        - Include **`# TODO:` comments** for every part the user must fill in.
        - Do NOT give a complete solution.
        - Add a note that the user must write the actual logic themselves.
        - Keep the code short and directly relevant to the issue.

        Example format:
        ```python
        def fix_problem(param):
            # TODO: Understand what 'param' does and implement the core logic
            pass
        End with a question asking them to try filling the TODOs.
        """,
        messages,
    )
    new_count = code_assist_count + 1
    logger.info(f"Code assist provided. Count now {new_count}/{MAX_ASSISTS}")

    return {
        **state,
        "messages": _assistant(messages, reply),
        "code_assist_count": new_count,
        "conversation_phase": "guidance",
    }
