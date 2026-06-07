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
import re

from core.models import Repository
from core.services.skills import SkillExtractor
from core.services.token_rotator import TokenRotator
from django.conf import settings
from pydantic import SecretStr

from ..graph_loader import load_engine_for_repo
from .state import AgentState
from .tools import fetch_code_snippet, fetch_repo_skills

logger = logging.getLogger(__name__)
_groq_rotator = TokenRotator(settings.GROQ_API_KEYS)


def get_llm():
    from langchain_groq import ChatGroq

    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=SecretStr(_groq_rotator.next()),
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


def _last_user_message(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _assistant(messages: list[dict], text: str) -> list[dict]:
    """Append an assistant message and return the updated list."""
    return [*messages, {"role": "assistant", "content": text}]


def _extract_file_paths(text: str) -> list[str]:
    """
    Naively extract potential file paths from issue title/body.
    Looks for common extensions: .py, .js, .ts, .java, .go, .rs, .cpp, .c, .h, .html, .css, etc.
    Returns a list of unique matches.
    """
    extensions = r"\.(py|js|ts|java|go|rs|cpp|c|h|html|css|json|xml|yaml|yml|md|txt)"
    pattern = r"\b[\w/\-\.]+" + extensions + r"\b"
    matches = re.findall(pattern, text, re.IGNORECASE)
    full_matches = [m[0] + m[1] for m in matches if m[0] and m[1]]
    return list(set(full_matches))[:3]


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

    if not state.get("user_skills"):
        if len(messages) <= 1:
            repo_skills = fetch_repo_skills.invoke(repo_url)
            reply = llm_respond(
                f"""
                The repository uses these skills most: {repo_skills}
                Ask the user naturally about their experience with these specific skills.
                Do NOT ask them to rate on a numeric scale.
                Example: 'I see this repo uses Python and REST APIs — how comfortable are you with those?'
                ONE question only.
                """,
                messages,
            )
            return {
                **state,
                "messages": _assistant(messages, reply),
                "conversation_phase": "onboarding",
            }

        skill_extractor = SkillExtractor()

        last_user_msg = _last_user_message(messages)
        extracted_skill_names = skill_extractor.extract(last_user_msg)

        repo = Repository.objects.get(id=state["repo_id"])
        repo_skills_set = set(repo.skills_found)

        valid_skills = [s for s in extracted_skill_names if s in repo_skills_set]

        if valid_skills:
            skills = [{"skill": s, "band": "intermediate"} for s in valid_skills]
            reply = "Great! Let me find the best issues for you — one moment."
            return {
                **state,
                "user_skills": skills,
                "messages": _assistant(messages, reply),
                "conversation_phase": "analysis",
            }
        else:
            reply = llm_respond(
                "The user hasn't mentioned any clear skills from our skill list. Ask them to list specific technologies (e.g., 'React', 'JavaScript', 'Python'). ONE short question.",
                messages,
            )
            return {
                **state,
                "messages": _assistant(messages, reply),
                "conversation_phase": "onboarding",
            }
    return {**state, "conversation_phase": "analysis"}


def issue_analysis_node(state: AgentState) -> AgentState:
    messages = state.get("messages") or []
    _ = state.get("repo_id")
    selected = state.get("selected_issue")
    recommendations = state.get("recommendations")

    if selected:
        return {**state, "conversation_phase": "guidance"}

    if not recommendations:
        reply = (
            "I'm waiting for the issue recommendations to be prepared. "
            "Please try again in a moment."
        )
        return {
            **state,
            "messages": _assistant(messages, reply),
            "conversation_phase": "analysis",
        }

    if len(messages) == 0 or messages[-1].get("role") != "assistant":
        reply = (
            "I've found several issues that match your skills. "
            "Please select one from the list above to continue."
        )
        return {
            **state,
            "messages": _assistant(messages, reply),
            "conversation_phase": "analysis",
        }

    return {**state, "conversation_phase": "analysis"}


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
    repo_id = state.get("repo_id")
    user_skills_list = [s["skill"] for s in (state.get("user_skills") or [])]
    issue_skills = selected_issue.get("skills") or []

    last = _last_user_message(messages)

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

        stuck = state.get("stuck_counter", 0) + 1
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

    guidelines_context = ""
    try:
        repo = Repository.objects.get(id=repo_id)
        guidelines_context = repo.contributing_guidelines
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
    repo_id = state.get("repo_id")
    user_approach = state.get("user_approach") or ""

    guidelines = "No CONTRIBUTING.md found."
    try:
        repo = Repository.objects.get(id=repo_id)
        guidelines = repo.contributing_guidelines
    except Exception as exc:
        logger.warning("Could not fetch guidelines: %s", exc)

    novelty = 1.0
    if user_approach and repo_id:
        try:
            engine = load_engine_for_repo(repo_id)
            novelty = engine.graph.novelty_score(
                user_approach, selected_issue.get("id", "")
            )
        except Exception as exc:
            logger.warning("Novelty score failed: %s", exc)

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
    If possible, fetches a relevant code snippet from the repo as context.
    """
    messages = state.get("messages") or []
    selected_issue = state.get("selected_issue") or {}
    repo_url = state.get("repo_url", "")
    repo_id = state.get("repo_id")
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

    context_snippet = ""
    issue_title = selected_issue.get("title", "")
    issue_summary = selected_issue.get("summary", "")
    issue_text = f"{issue_title} {issue_summary}"
    file_paths = _extract_file_paths(issue_text)

    if file_paths:
        for file_path in file_paths:
            snippet = fetch_code_snippet.invoke(
                {"repo_url": repo_url, "file_path": file_path}
            )
            if (
                snippet
                and not snippet.startswith("Could not fetch")
                and not snippet.startswith("Error")
            ):
                context_snippet = (
                    f"\n**Relevant code from `{file_path}`:**\n```\n{snippet}\n```\n"
                )
                break

    guidelines_context = ""
    try:
        repo = Repository.objects.get(id=repo_id)
        guidelines_context = repo.contributing_guidelines[:1000]
    except Exception as e:
        logger.warning("Could not fetch guidelines for repo %s: %s", repo_id, e)

    prompt = f"""
    Issue: {json.dumps(selected_issue)}
    Repository: {repo_url}
    User's previous understanding: {state.get("understanding_score", "unknown")}
    Contributing Guidelines (excerpt): {guidelines_context[:500]}
    {context_snippet}

    The user is stuck. Provide a **boilerplate code snippet** that outlines the structure needed.
    - Include **`# TODO:` comments** for every part the user must fill in.
    - Use the provided code snippet (if any) to make the boilerplate relevant to the actual codebase.
    - Do NOT give a complete solution.
    - Add a note that the user must write the actual logic themselves.
    - Keep the code short and directly relevant to the issue.

    Example format:
    ```python
    def fix_problem(param):
        # TODO: Understand what 'param' does and implement the core logic
        pass
    End with a question asking them to try filling the TODOs.
    """
    reply = llm_respond(prompt, messages)
    new_count = code_assist_count + 1
    logger.info(f"Code assist provided. Count now {new_count}/{MAX_ASSISTS}")

    return {
        **state,
        "messages": _assistant(messages, reply),
        "code_assist_count": new_count,
        "conversation_phase": "guidance",
    }
