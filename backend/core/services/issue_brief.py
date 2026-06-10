# backend/core/services/issue_brief.py
import re


def _clean_body(text: str, limit: int = 280) -> str:
    if not text:
        return ""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", "", text)
    text = re.sub(r"[#*`>]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def _infer_action(title: str, body: str, labels: list[str]) -> str:
    text = f"{title} {body}".lower()
    label_lower = {label.lower() for label in labels}

    if "good first issue" in label_lower or "documentation" in label_lower:
        return (
            "Read the issue carefully, make the small scoped change described, "
            "and open a PR that references this issue number."
        )
    if "deps" in text or "dependabot" in text or "chore(deps)" in text:
        return (
            "Find the workflow or config file that pins the dependency, bump the "
            "version as requested, and verify CI still passes."
        )
    if "github-script" in text or "github actions" in text or "workflow" in text:
        return (
            "Locate the GitHub Actions workflow file, update the action version or "
            "syntax, and confirm the workflow still runs successfully."
        )
    if "fix" in text or "bug" in text:
        return (
            "Reproduce the bug if possible, trace it to the relevant module, implement "
            "a minimal fix, and add or update tests."
        )
    if "test" in text or "spec" in text:
        return (
            "Identify the behaviour that needs coverage, add focused tests, and ensure "
            "the full test suite passes."
        )
    if "refactor" in text:
        return (
            "Understand the current implementation, refactor without changing behaviour, "
            "and keep existing tests green."
        )
    return (
        "Review the issue description and linked discussion, identify the files involved, "
        "implement the requested change, and submit a PR referencing this issue."
    )


def build_issue_brief(issue: dict) -> dict:
    """Return human-readable about + action fields for an issue."""
    title = issue.get("title", "")
    summary = _clean_body(issue.get("summary") or "")
    labels = issue.get("labels") or []

    if summary:
        about = summary
    else:
        about = f"This issue tracks: {title}"

    return {
        "about": about,
        "action": _infer_action(title, issue.get("summary") or "", labels),
    }
