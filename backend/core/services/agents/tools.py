# core/services/agents/tools.py

import requests as http_requests
from django.conf import settings
from langchain.tools import tool

from ..skills import SkillExtractor

extractor = SkillExtractor()


@tool
def fetch_repo_skills(repo_url: str) -> list:
    """Fetch top skills needed in this repo"""
    token = getattr(settings, "GITHUB_TOKEN", "")
    owner, repo = repo_url.rstrip("/").split("/")[-2:]
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    headers = {"Authorization": f"token {token}"}
    params = {"state": "open", "per_page": 30}

    response = http_requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        return []

    issues = response.json()
    all_skills = []
    for issue in issues:
        if "pull_request" in issue:
            continue
        text = issue.get("title", "") + " " + (issue.get("body") or "")
        all_skills.extend(extractor.extract(text))

    from collections import Counter

    return [s for s, _ in Counter(all_skills).most_common(6)]


@tool
def fetch_contributing_guidelines(repo_url: str) -> str:
    """Fetch CONTRIBUTING.md from the repo"""
    owner, repo = repo_url.rstrip("/").split("/")[-2:]
    urls = [
        f"https://raw.githubusercontent.com/{owner}/{repo}/main/CONTRIBUTING.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/master/CONTRIBUTING.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/main/.github/CONTRIBUTING.md",
    ]
    for url in urls:
        response = http_requests.get(url)
        if response.status_code == 200:
            return response.text[:3000]
    return "No CONTRIBUTING.md found for this repo."
