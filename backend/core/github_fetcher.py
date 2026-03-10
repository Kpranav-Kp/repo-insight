"""
github_fetcher.py
=================

Module 1: Fetch real issues + PRs from GitHub repository
"""

import requests


class GitHubFetcher:

    def __init__(self, repo_url):
        """
        repo_url example:
        https://github.com/pallets/flask
        """
        self.repo_url = repo_url
        self.owner, self.repo = repo_url.rstrip("/").split("/")[-2:]

        self.base_api = f"https://api.github.com/repos/{self.owner}/{self.repo}"

    # ---------------------------------------------------
    # Fetch Issues
    # ---------------------------------------------------

    def fetch_issues(self, limit=50):

        url = f"{self.base_api}/issues"

        params = {
            "state": "open",
            "per_page": limit
        }

        response = requests.get(url, params=params)

        if response.status_code != 200:
            raise Exception("GitHub API request failed")

        issues = response.json()

        results = []

        for issue in issues:

            # skip pull requests
            if issue.get("pull_request") is not None:
                continue

            results.append({
                "id": str(issue["number"]),
                "title": issue["title"],
                "summary": issue.get("body", "")[:500]
            })

        print(f"Fetched {len(results)} issues")

        return results

    # ---------------------------------------------------
    # Fetch Pull Requests
    # ---------------------------------------------------

    def fetch_prs(self, limit=20):

        url = f"{self.base_api}/pulls"

        params = {
            "state": "closed",
            "per_page": limit
        }

        response = requests.get(url, params=params)

        if response.status_code != 200:
            raise Exception("GitHub API request failed")

        prs = response.json()

        results = []

        for pr in prs:
            results.append({
                "id": str(pr["number"]),
                "title": pr["title"],
                "issue_id": str(pr["number"])
            })

        return results