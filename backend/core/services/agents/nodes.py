# backend/core/services/agents/nodes.py

from django.conf import settings
from langchain_openai import ChatOpenAI

from ..graph_loader import load_engine_for_repo
from .state import AgentState
from .tools import (
    fetch_contributing_guidelines,
    fetch_repo_skills,
)

# ── LLM SETUP ────────────────────────────────────────────────────────────────


def get_llm():
    return ChatOpenAI(
        model=settings.OPENROUTER_MODEL,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
    )


# ── HELPER ────────────────────────────────────────────────────────────────────


def llm_respond(system_prompt: str, messages: list) -> str:
    """Send messages to LLM and get response text back."""
    llm = get_llm()
    response = llm.invoke([{"role": "system", "content": system_prompt}, *messages])
    return response.content


# ── AGENT 1: ONBOARDING ───────────────────────────────────────────────────────


def onboarding_node(state: AgentState) -> AgentState:
    """
    Collects 3 things:
      1. repo_url
      2. user_skills with experience band
      3. intent (learn or vibe)

    Asks one question at a time.
    Moves to analysis phase when all 3 collected.
    """
    messages = state.get("messages", [])
    last_message = messages[-1]["content"] if messages else ""

    # ── collect repo_url ──
    if not state.get("repo_url"):
        response = llm_respond(
            """
            You are helping a developer find open source issues to contribute to.
            Ask them for the GitHub repository URL they want to contribute to.
            Be friendly and conversational. One question only.
            """,
            messages,
        )
        messages.append({"role": "assistant", "content": response})
        return {**state, "messages": messages, "conversation_phase": "onboarding"}

    # ── collect user_skills ──
    if not state.get("user_skills"):
        repo_skills = fetch_repo_skills.invoke(state["repo_url"])
        response = llm_respond(
            f"""
            The repository needs these skills: {repo_skills}
            Ask the user specifically about their experience with these skills.
            Infer their level from how they describe themselves naturally.
            Do NOT ask them to rate themselves on a scale.
            Example: 'I see this repo uses Python and REST APIs a lot.
            How comfortable are you with those?'
            One question only.
            """,
            messages,
        )
        messages.append({"role": "assistant", "content": response})
        return {**state, "messages": messages, "conversation_phase": "onboarding"}

    # ── parse skills from conversation ──
    if not state.get("user_skills") or len(state["user_skills"]) == 0:
        parsed = llm_respond(
            """
            Based on this conversation, extract the user's skills and experience.
            Return ONLY a JSON list like:
            [{"skill": "python", "band": "intermediate"},
             {"skill": "sql", "band": "beginner"}]

            Bands must be: beginner, intermediate, or comfortable.
            Nothing else in your response — just the JSON list.
            """,
            messages,
        )
        import json

        try:
            skills = json.loads(parsed.strip())
        except:
            skills = []
        state = {**state, "user_skills": skills}

    # ── collect intent ──
    if not state.get("intent"):
        response = llm_respond(
            """
            Ask the user how they want to approach this contribution:
            Option A: Learn mode — they want to understand deeply,
                      you will guide them without giving code
            Option B: Vibe mode — they just want the solution quickly

            Be casual. One question only.
            """,
            messages,
        )
        messages.append({"role": "assistant", "content": response})
        return {**state, "messages": messages, "conversation_phase": "onboarding"}

    # ── parse intent ──
    intent_parsed = (
        llm_respond(
            """
        Based on the last message, did the user choose:
        - learn mode (wants to understand)
        - vibe mode (wants solution fast)

        Reply with ONLY one word: learn OR vibe
        """,
            messages,
        )
        .strip()
        .lower()
    )

    intent = "learn" if "learn" in intent_parsed else "vibe"

    messages.append(
        {"role": "assistant", "content": "Great! Let me find the best issues for you."}
    )

    return {
        **state,
        "messages": messages,
        "intent": intent,
        "conversation_phase": "analysis",
    }


# ── AGENT 2: ISSUE ANALYSIS ───────────────────────────────────────────────────


def issue_analysis_node(state: AgentState) -> AgentState:
    """
    Finds matching issues from graph.
    Explains them in plain language.
    Waits for user to pick one.
    """
    messages = state.get("messages", [])
    repo_id = state.get("repo_id")
    skills = state.get("user_skills", [])
    skill_names = [s["skill"] for s in skills]

    # ── find matching issues ──
    if not state.get("recommendations"):
        engine = load_engine_for_repo(repo_id)
        raw_results = engine.recommend(skill_names, top_k=5)

        # filter by experience band
        experience = skills[0]["band"] if skills else "beginner"
        if experience == "beginner":
            raw_results = [r for r in raw_results if len(r.get("skills", [])) <= 4]

        # check novelty
        flagged = []
        for r in raw_results:
            score = engine.graph.novelty_score("", r["id"])
            r["novelty"] = round(score, 2)
            if score < 0.5:
                r["already_tried"] = True
            flagged.append(r)

        state = {**state, "recommendations": flagged}

    # ── check if user already picked an issue ──
    if state.get("selected_issue"):
        return {
            **state,
            "conversation_phase": "guidance"
            if state.get("intent") == "learn"
            else "review",
        }

    # ── explain issues in plain language ──
    recommendations = state["recommendations"]
    response = llm_respond(
        f"""
        You are helping a developer pick a GitHub issue to work on.
        Here are the top matching issues: {recommendations}

        For each issue explain in plain simple language:
        - What the issue is asking for
        - What skills are needed
        - How complex it roughly is
        - If novelty < 0.5 warn: 'Note: A similar approach was already tried'

        Present maximum 3 issues.
        End by asking which one they want to work on.
        DO NOT give any code.
        """,
        messages,
    )

    messages.append({"role": "assistant", "content": response})

    # ── parse which issue user picked ──
    if len(messages) > 1:
        last_user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break

        if last_user_msg:
            picked = llm_respond(
                f"""
                The user said: "{last_user_msg}"
                Available issues: {recommendations}

                Did the user pick one of these issues?
                If yes reply with ONLY the issue id number.
                If no reply with: none
                """,
                messages,
            ).strip()

            if picked != "none":
                selected = next(
                    (r for r in recommendations if str(r["id"]) == picked), None
                )
                if selected:
                    return {
                        **state,
                        "messages": messages,
                        "selected_issue": selected,
                        "conversation_phase": "guidance"
                        if state.get("intent") == "learn"
                        else "review",
                    }

    return {**state, "messages": messages, "conversation_phase": "analysis"}


# ── AGENT 3: GUIDANCE ─────────────────────────────────────────────────────────


def guidance_node(state: AgentState) -> AgentState:
    """
    Runs in a LOOP.
    Asks user questions about the issue.
    Evaluates understanding.
    Never gives code.
    Only moves forward when understanding = SUFFICIENT.
    """
    messages = state.get("messages", [])
    selected_issue = state.get("selected_issue", {})
    repo_url = state.get("repo_url", "")

    # ── get last user message ──
    last_user_msg = ""
    for m in reversed(messages):
        if m["role"] == "user":
            last_user_msg = m["content"]
            break

    # ── evaluate understanding if user said something ──
    if last_user_msg:
        evaluation = llm_respond(
            f"""
            Issue context: {selected_issue}
            User's answer: "{last_user_msg}"

            Evaluate if this answer shows SPECIFIC understanding of the issue.
            A SUFFICIENT answer:
              - references the actual problem described
              - shows understanding of why it happens
              - suggests a general direction (not code)

            An INSUFFICIENT answer:
              - is vague or generic
              - could apply to any issue
              - is clearly AI generated text

            Reply with ONLY: SUFFICIENT or INSUFFICIENT
            """,
            messages,
        ).strip()

        state = {**state, "understanding_score": evaluation}

        if evaluation == "SUFFICIENT":
            # check for vibe coding
            vibe_check = llm_respond(
                f"""
                User's explanation: "{last_user_msg}"

                Does this read like:
                A) A genuine developer response with specific details
                B) Generic AI-generated text without specific details

                Reply ONLY: genuine OR vibe_coded
                """,
                messages,
            ).strip()

            if "genuine" in vibe_check:
                messages.append(
                    {
                        "role": "assistant",
                        "content": "Great understanding! Let me check your approach against the repo guidelines.",
                    }
                )
                return {**state, "messages": messages, "conversation_phase": "review"}
            else:
                # vibe coded detected
                response = llm_respond(
                    f"""
                    The user's answer seems AI-generated or too generic.
                    Issue: {selected_issue}

                    Ask them a very specific follow-up question that requires
                    them to reference the actual code or file involved.
                    DO NOT give any code or hints.
                    """,
                    messages,
                )
                messages.append({"role": "assistant", "content": response})
                return {**state, "messages": messages, "conversation_phase": "guidance"}

    # ── check for skill gap and generate learning path ──
    user_skills = [s["skill"] for s in state.get("user_skills", [])]
    issue_skills = selected_issue.get("skills", [])
    gaps = [s for s in issue_skills if s not in user_skills]

    if gaps and not last_user_msg:
        learning_path = llm_respond(
            f"""
            Developer knows: {user_skills}
            Issue needs    : {issue_skills}
            Skill gaps     : {gaps}

            Create a learning path for the gaps.
            For each gap provide:
              - One concept to understand
              - One resource to read (real URL if you know it)
              - One small exercise to try
              - Estimated time

            NO code. NO solutions.
            End with: 'Come back after going through these!'
            """,
            messages,
        )
        messages.append({"role": "assistant", "content": learning_path})
        return {**state, "messages": messages, "conversation_phase": "guidance"}

    # ── ask first understanding question ──
    relevant_docs = fetch_contributing_guidelines.invoke(repo_url)
    response = llm_respond(
        f"""
        Issue: {selected_issue}
        Repo : {repo_url}

        Ask the user ONE specific question to check their understanding.
        The question should make them think about:
          - What is actually causing this issue
          - Where in the codebase it might be

        DO NOT give any code.
        DO NOT give hints about the answer.
        Point them to where to look, not what to write.
        """,
        messages,
    )
    messages.append({"role": "assistant", "content": response})
    return {**state, "messages": messages, "conversation_phase": "guidance"}


# ── AGENT 4: REVIEW ───────────────────────────────────────────────────────────


def review_node(state: AgentState) -> AgentState:
    """
    Validates user's approach before they write code.
    Checks against repo's contributing guidelines.
    Checks novelty score.
    Generates PR outline template (not the code).
    """
    messages = state.get("messages", [])
    selected_issue = state.get("selected_issue", {})
    repo_url = state.get("repo_url", "")
    repo_id = state.get("repo_id")
    user_approach = state.get("user_approach", "")

    # ── get contributing guidelines ──
    guidelines = fetch_contributing_guidelines.invoke(repo_url)

    # ── check novelty of user's approach ──
    novelty = 1.0
    if user_approach and repo_id:
        engine = load_engine_for_repo(repo_id)
        novelty = engine.graph.novelty_score(
            user_approach, selected_issue.get("id", "")
        )

    # ── generate final checklist + PR outline ──
    response = llm_respond(
        f"""
        Issue         : {selected_issue}
        User's approach: {user_approach or "not yet described"}
        Guidelines    : {guidelines}
        Novelty score : {novelty} (below 0.5 means similar approach was tried)

        Produce a final output with these sections:

        1. APPROACH REVIEW
           - What the user is doing right
           - What might get the PR rejected
           - What to clarify with maintainers first

        2. NOVELTY CHECK
           - If novelty < 0.5: warn that similar approach was tried,
             summarise what was attempted before
           - If novelty >= 0.5: confirm the approach looks original

        3. PR OUTLINE TEMPLATE
           - Correct format for this specific repo
           - Include: issue reference, change summary,
             testing notes, checklist items
           - Leave blanks for user to fill in themselves
           - DO NOT write the actual code changes

        Remember: guide them, do not write code for them.
        """,
        messages,
    )

    messages.append({"role": "assistant", "content": response})

    return {**state, "messages": messages, "conversation_phase": "complete"}
