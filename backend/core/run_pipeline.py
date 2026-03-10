"""
recommendation_agent.py
=======================

Module 5: Agents that query the semantic graph
"""
print("Script started...")
from github_fetcher import GitHubFetcher
from skill_extractor import SkillExtractor
from semantic_graph import SemanticGraph


class RecommendationAgent:

    def __init__(self, repo_url):

        self.repo_url = repo_url

        self.fetcher = GitHubFetcher(repo_url)
        self.extractor = SkillExtractor()
        self.graph = SemanticGraph()

    # ---------------------------------------------------
    # Build Graph Pipeline
    # ---------------------------------------------------

    def build_graph(self):

        print("Fetching issues...")

        issues = self.fetcher.fetch_issues()

        for issue in issues:

            text = issue["title"] + " " + issue["summary"]

            skills = self.extractor.extract(text)

            issue["skills"] = skills

            self.graph.add_issue(issue)

        print("Fetching PRs...")

        prs = self.fetcher.fetch_prs()

        for pr in prs:
            self.graph.add_pr(pr)

        print("Building graph...")

        self.graph.build_edges()

        print("Graph stats:", self.graph.stats())

    # ---------------------------------------------------
    # Recommend Issues
    # ---------------------------------------------------

    def recommend(self, user_skills):

        print("\nFinding matching issues...\n")

        results = self.graph.skill_to_issue(user_skills)

        for r in results:
            print(f"[{r['score']}] {r['title']}")

        return results


# ---------------------------------------------------
# Run Agent
# ---------------------------------------------------

if __name__ == "__main__":

    repo_url = input("Enter GitHub repo URL: ")

    agent = RecommendationAgent(repo_url)

    agent.build_graph()

    user_skills = input("\nEnter your skills (comma separated): ").split(",")

    user_skills = [s.strip() for s in user_skills]

    agent.recommend(user_skills)