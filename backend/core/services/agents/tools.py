# backend/core/services/agents/tools.py
from collections import Counter

import requests as http_requests
from django.conf import settings
from langchain.tools import tool

from ..skills import SkillExtractor

extractor = SkillExtractor()


@tool
def fetch_repo_skills(repo_url: str) -> list[str]:
    """Fetch the top skills needed to contribute to this GitHub repository."""
    token = getattr(settings, "GITHUB_TOKEN", "")
    try:
        owner, repo = repo_url.rstrip("/").split("/")[-2:]
    except ValueError:
        return []

    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    headers = {"Authorization": f"token {token}"} if token else {}
    params = {"state": "open", "per_page": 30}

    try:
        response = http_requests.get(url, headers=headers, params=params, timeout=10)
    except http_requests.RequestException:
        return []

    if response.status_code != 200:
        return []

    all_skills: list[str] = []
    for issue in response.json():
        if "pull_request" in issue:
            continue
        text = issue.get("title", "") + " " + (issue.get("body") or "")
        all_skills.extend(extractor.extract(text))

    return [s for s, _ in Counter(all_skills).most_common(6)]


@tool
def fetch_contributing_guidelines(repo_url: str) -> str:
    """Fetch the CONTRIBUTING.md from a GitHub repository."""
    try:
        owner, repo = repo_url.rstrip("/").split("/")[-2:]
    except ValueError:
        return "Could not parse repository URL."

    candidates = [
        f"https://raw.githubusercontent.com/{owner}/{repo}/main/CONTRIBUTING.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/master/CONTRIBUTING.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/main/.github/CONTRIBUTING.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/master/.github/CONTRIBUTING.md",
    ]

    for url in candidates:
        try:
            response = http_requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.text[:3000]
        except http_requests.RequestException:
            continue

    return "No CONTRIBUTING.md found for this repository."


@tool
def fetch_code_snippet(
    repo_url: str, file_path: str, line_start: int = 0, line_end: int = 0
) -> str:
    """
    Fetch a specific code file (or a range of lines) from the GitHub repository.
    Useful for pointing the user to relevant functions or modules.
    """
    token = getattr(settings, "GITHUB_TOKEN", "")
    try:
        owner, repo = repo_url.rstrip("/").split("/")[-2:]
    except ValueError:
        return "Invalid repository URL."

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    headers = {"Authorization": f"token {token}"} if token else {}

    try:
        response = http_requests.get(api_url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"Could not fetch file: {response.status_code}"
        data = response.json()
        if "content" not in data:
            return "File content not found."
        import base64

        content = base64.b64decode(data["content"]).decode("utf-8")
        lines = content.splitlines()
        if line_end > 0:
            lines = lines[line_start:line_end]
        elif line_start > 0:
            lines = lines[line_start - 1 : line_start + 10]
        return "\n".join(lines[:50])
    except Exception as e:
        return f"Error fetching snippet: {str(e)}"
