# github.py
import requests
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type


class GitHubError(Exception):
    pass


class RateLimitError(GitHubError):
    pass


@dataclass
class IssueData:
    github_id: str
    number: int
    title: str
    body: str
    state: str
    labels: List[str]
    skills: List[str] = field(default_factory=list)


@dataclass
class PRData:
    github_id: str
    number: int
    title: str
    body: str
    state: str
    linked_issue_numbers: List[int]


class GitHubClient():

    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "RepoInsight/1.0",
        })
        if token:
            self.session.headers["Authorization"] = f"token {token}"

    def _check_rate_limit(self, response):
        remaining = int(response.headers.get("X-RateLimit-Remaining", 1))
        reset_at = int(response.headers.get("X-RateLimit-Reset", 0))

        if response.status_code == 403 or remaining == 0:
            raise RateLimitError(
                f"Rate limit exceeded. Resets at: {reset_at}"
            )

    def _parse_repo_url(self, url: str) -> tuple:
        url = url.rstrip("/").removesuffix(".git")
        match = re.search(r"github\.com/([^/]+)/([^/]+)", url)
        if not match:
            raise ValueError(f"Invalid GitHub URL: {url}")
        return match.group(1), match.group(2)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_not_exception_type(RateLimitError), reraise=True)
    def fetch_issues(self, repo_url: str, limit: int = 50, state: str = "open") -> List[IssueData]:
        owner, repo = self._parse_repo_url(repo_url)
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"

        issues = []
        page = 1
        per_page = min(limit, 100)
        while len(issues) < limit:
            params = {"state": state, "per_page": per_page, "page": page}
            response = self.session.get(url, params=params, timeout=30)
            self._check_rate_limit(response)

            if response.status_code != 200:
                raise GitHubError(f"GitHub API error: {response.status_code}")

            batch = response.json()
            if not batch:
                break  

            for item in batch:
                if "pull_request" in item:
                    continue
                if len(issues) >= limit:
                    break
                issues.append(IssueData(
                    github_id=str(item["id"]),
                    number=item["number"],
                    title=item["title"],
                    body=item.get("body", "") or "",
                    state=item["state"],
                    labels=[label["name"] for label in item.get("labels", [])],
                ))

            page += 1

        return issues

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_not_exception_type(RateLimitError), reraise=True)
    def fetch_pull_requests(self, repo_url: str, limit: int = 20, state: str = "closed") -> List[PRData]:
        owner, repo = self._parse_repo_url(repo_url)
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"

        prs = []
        page = 1
        per_page = min(limit, 100)

        while len(prs) < limit:
        
            params = {"state": state, "per_page": per_page, "page": page}
            response = self.session.get(url, params=params, timeout=30)

            self._check_rate_limit(response)

            if response.status_code != 200:
                raise GitHubError(f"GitHub API error: {response.status_code}")
            
            batch = response.json()
            if not batch:
                break 

            
            for item in batch:
                text = f"{item.get('title', '')} {item.get('body', '')}"
                linked_issues = self._extract_linked_issues(text)
                
                if len(prs) >= limit:
                    break

                if not item.get("merged_at"):
                    continue

                if item.get("merged_at"):
                    pr_state = "merged"
                else:
                    pr_state = item["state"]

                prs.append(PRData(
                    github_id=str(item["id"]),
                    number=item["number"],
                    title=item["title"],
                    body=item.get("body", "") or "",
                    state=pr_state,
                    linked_issue_numbers=linked_issues,
                ))
            page += 1

        return prs

    def _extract_linked_issues(self, text: str) -> List[int]:
        if not text:
            return []
        
        patterns = [
            r"(?:fixes|closes|resolves|refs?)\s*#(\d+)",
        ]
        
        issues = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            issues.update(int(m) for m in matches)
        
        return sorted(list(issues))