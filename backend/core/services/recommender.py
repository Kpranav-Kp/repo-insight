# backend/core/services/recommender.py
import json
import os

from django.conf import settings

from .github import GitHubClient, GitHubError, RateLimitError
from .semantic_graph import SemanticGraph
from .skills import SkillExtractor, extract_issue_metadata


class RecommendationEngine:
    def __init__(self, github_token: str | None = None):
        self.github = GitHubClient(token=github_token)
        self.skill_extractor = SkillExtractor()
        model_path = getattr(settings, "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        self.graph = SemanticGraph(model_name=model_path)
        self._is_built = False

    def build_from_repository(self, repo_url: str) -> dict:
        """
        Fetch issues + PRs from repo, populate the semantic graph,
        and build all edges.

        Returns a summary dict for the caller (e.g. Celery task).
        Raises RuntimeError on GitHub failures.
        """
        try:
            issues = self.github.fetch_issues(
                repo_url, limit=settings.GITHUB_ISSUE_LIMIT
            )
        except RateLimitError as e:
            raise RuntimeError(f"GitHub rate limit hit: {e}") from e
        except GitHubError as e:
            raise RuntimeError(f"Failed to fetch issues: {e}") from e

        try:
            prs = self.github.fetch_pull_requests(
                repo_url, limit=settings.GITHUB_PR_LIMIT
            )
        except RateLimitError as e:
            raise RuntimeError(f"GitHub rate limit hit: {e}") from e
        except GitHubError as e:
            raise RuntimeError(f"Failed to fetch PRs: {e}") from e

        all_skills = set()

        for issue in issues:
            metadata = extract_issue_metadata(issue.title, issue.body, issue.labels)
            skills = metadata["skills"]
            difficulty = metadata["difficulty"]
            all_skills.update(skills)
            self.graph.add_issue(
                {
                    "id": str(issue.number),
                    "title": issue.title,
                    "summary": issue.body[:500],
                    "skills": skills,
                    "labels": issue.labels,
                    "state": issue.state,
                    "difficulty": difficulty,  # new field
                }
            )
        for pr in prs:
            if not pr.linked_issue_numbers:
                self.graph.add_pr(
                    {
                        "id": str(pr.number),
                        "title": pr.title,
                        "issue_id": None,
                        "created_at": pr.created_at,
                    }
                )
            else:
                for linked_issue_id in pr.linked_issue_numbers:
                    self.graph.add_pr(
                        {
                            "id": str(pr.number),
                            "title": pr.title,
                            "issue_id": str(linked_issue_id),
                            "created_at": pr.created_at,
                        }
                    )

        self.graph.build_edges()
        self._is_built = True

        return {
            "repository_url": repo_url,
            "issues_indexed": len(issues),
            "prs_indexed": len(prs),
            "skills_found": sorted(all_skills),
            "graph_stats": self.graph.stats(),
        }

    def recommend(self, user_skills: list[str], top_k: int = 5) -> list[dict]:
        """
        Metadata-driven recommendation.
        1. Use FAISS to retrieve top (top_k * 2) candidate issues.
        2. Re-rank each candidate using:
           final_score = 0.6 * skill_overlap + 0.2 * difficulty_score + 0.2 * label_bonus
        3. Return top_k issues sorted by final_score.
        """
        if not self._is_built:
            raise RuntimeError("Graph not built. Call build_from_repository() first.")

        candidates = self.graph.skill_to_issue(user_skills, top_k=top_k * 2)

        if not candidates:
            return []

        user_skills_set = set(s.lower() for s in user_skills)

        scored = []
        for cand in candidates:
            issue_id = cand["id"]
            issue_skills = set(cand.get("skills", []))
            if issue_skills:
                skill_overlap = len(user_skills_set & issue_skills) / len(issue_skills)
            else:
                skill_overlap = 0.0

            difficulty_score = self.graph.get_issue_difficulty_score(issue_id)
            label_bonus = self.graph.get_issue_label_bonus(issue_id)

            final_score = (
                0.6 * skill_overlap + 0.2 * difficulty_score + 0.2 * label_bonus
            )

            cand["skill_overlap"] = sorted(user_skills_set & issue_skills)
            cand["match_score"] = round(skill_overlap, 4)
            cand["difficulty_score"] = difficulty_score
            cand["label_bonus"] = label_bonus
            cand["combined_score"] = round(final_score, 4)

            scored.append(cand)

        # Sort by combined_score descending
        scored.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored[:top_k]

    def check_duplicate(self, issue_text: str) -> tuple:
        """
        Check whether a given text is a near-duplicate of an indexed issue.
        Delegates to SemanticGraph which owns the DEDUP_THRESHOLD.

        Returns (True, matched_issue_dict) or (False, None).
        """
        if not self._is_built:
            raise RuntimeError("Graph not built. Call build_from_repository() first.")
        return self.graph.is_duplicate_issue(issue_text)

    def save_index(self, directory: str):
        """
        Persist the FAISS index + metadata to disk.
        Note: graph edges are in-memory and must be rebuilt on load.
        """
        if not self._is_built:
            raise RuntimeError("Nothing to save. Build the graph first.")

        self.graph.issues._service.save(os.path.join(directory, "issues"))
        self.graph.skills._service.save(os.path.join(directory, "skills"))
        self.graph.prs._service.save(os.path.join(directory, "prs"))

        with open(os.path.join(directory, "edges.json"), "w") as f:
            json.dump(self.graph.adj.edges, f)

    def load_index(self, directory: str):
        self.graph.issues._service.load(os.path.join(directory, "issues"))
        self.graph.skills._service.load(os.path.join(directory, "skills"))
        self.graph.prs._service.load(os.path.join(directory, "prs"))

        with open(os.path.join(directory, "edges.json")) as f:
            self.graph.adj.edges = json.load(f)

        self.graph.issues.meta = list(self.graph.issues._service.metadata.values())
        self.graph.skills.meta = list(self.graph.skills._service.metadata.values())
        self.graph.prs.meta = list(self.graph.prs._service.metadata.values())

        self.graph.issues._id_to_idx = {
            str(m.get("id", "")): i for i, m in enumerate(self.graph.issues.meta)
        }
        self.graph.skills._id_to_idx = {
            str(m.get("id", "")): i for i, m in enumerate(self.graph.skills.meta)
        }
        self.graph.prs._id_to_idx = {
            str(m.get("id", "")): i for i, m in enumerate(self.graph.prs.meta)
        }
        self._is_built = True
