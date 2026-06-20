import json
import math
import os
from datetime import datetime

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
        self._repo_language: str | None = None

    def build_from_repository(self, repo_url: str) -> dict:
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

        try:
            languages = self.github.fetch_languages(repo_url)
        except (RateLimitError, GitHubError):
            languages = {}

        top_lang = (
            max(languages, key=lambda k: languages[k] or 0) if languages else None
        )
        self._repo_language = top_lang

        all_skills = set()

        for issue in issues:
            metadata = extract_issue_metadata(issue.title, issue.body, issue.labels)
            skills = metadata["skills"]
            difficulty = metadata["difficulty"]
            if top_lang and top_lang.lower() not in {s.lower() for s in skills}:
                skills.append(top_lang.lower())
            all_skills.update(skills)
            self.graph.add_issue(
                {
                    "id": str(issue.number),
                    "title": issue.title,
                    "summary": issue.body[:500],
                    "skills": skills,
                    "labels": issue.labels,
                    "state": issue.state,
                    "difficulty": difficulty,
                    "created_at": issue.created_at,
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

    def recommend(
        self,
        user_skills: list[str],
        top_k: int = 5,
        exclude_issue_ids: set[str] | None = None,
    ) -> list[dict]:
        if not self._is_built:
            raise RuntimeError("Graph not built. Call build_from_repository() first.")

        exclude_issue_ids = exclude_issue_ids or set()

        # Step 1: FAISS skill-to-issue retrieval (generous top_k)
        candidates = self.graph.skill_to_issue(user_skills, top_k=top_k * 3)
        if not candidates:
            return []

        # Step 2: Filter — open only & not resolved by merged PR & not claimed
        filtered = []
        resolved_ids = self._resolved_issue_ids()
        for cand in candidates:
            issue_id = cand["id"]
            if cand.get("state") != "open":
                continue
            if issue_id in resolved_ids:
                continue
            if issue_id in exclude_issue_ids:
                continue
            filtered.append(cand)

        if not filtered:
            return []

        # Step 3: Multi-hop expansion via ISSUE_ISSUE_SIM edges
        candidate_ids = {c["id"] for c in filtered}
        expanded_ids = set(candidate_ids)
        for cid in list(candidate_ids):
            connected = self.graph.get_connected_issues(cid)
            for conn in connected:
                if (
                    conn not in expanded_ids
                    and conn not in resolved_ids
                    and conn not in exclude_issue_ids
                ):
                    expanded_ids.add(conn)

        # Fetch metadata for expanded issues not already in filtered
        seen_ids = {c["id"] for c in filtered}
        if expanded_ids - seen_ids:
            extra = self.graph.get_issues_by_ids(expanded_ids - seen_ids)
            filtered.extend(extra)

        user_skills_set = set(s.lower() for s in user_skills)
        now = datetime.now()

        scored = []
        for cand in filtered:
            issue_id = cand["id"]
            issue_skills = set(s.lower() for s in cand.get("skills", []))
            semantic_score = float(cand.get("score", 0))

            overlap_skills = user_skills_set & issue_skills
            if issue_skills:
                skill_overlap = len(overlap_skills) / len(issue_skills)
                user_coverage = (
                    len(overlap_skills) / len(user_skills_set)
                    if user_skills_set
                    else 0.0
                )
            else:
                skill_overlap = 0.0
                user_coverage = 0.0

            difficulty_score = self.graph.get_issue_difficulty_score(issue_id)
            label_bonus = self.graph.get_issue_label_bonus(issue_id)
            normalized_label = label_bonus / 0.3 if label_bonus else 0.0

            # Recency: exp decay, 0 = very old, 1 = fresh
            created_str = cand.get("created_at", "")
            recency_score = 1.0
            if created_str:
                try:
                    dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                    months = (now - dt).days / 30.44
                    recency_score = max(0.0, math.exp(-0.08 * months))
                except (ValueError, TypeError):
                    pass

            skill_component = skill_overlap if issue_skills else semantic_score
            coverage_component = user_coverage if issue_skills else semantic_score * 0.5

            final_score = (
                0.25 * skill_component
                + 0.25 * semantic_score
                + 0.15 * coverage_component
                + 0.10 * difficulty_score
                + 0.10 * normalized_label
                + 0.15 * recency_score
            )

            cand["skill_overlap"] = sorted(overlap_skills)
            cand["match_score"] = round(
                skill_overlap if issue_skills else semantic_score, 4
            )
            cand["semantic_score"] = round(semantic_score, 4)
            cand["difficulty_score"] = difficulty_score
            cand["label_bonus"] = label_bonus
            cand["recency_score"] = round(recency_score, 4)
            cand["combined_score"] = round(final_score, 4)

            scored.append(cand)

        scored.sort(key=lambda x: x["combined_score"], reverse=True)

        # Step 4: Diversity re-ranking — demote clustered issues
        diverse = self._diversify(scored, top_k)
        return diverse[:top_k]

    def _resolved_issue_ids(self) -> set[str]:
        """Return set of issue IDs that have been resolved by a merged PR."""
        resolved = set()
        for edge in self.graph.adj.get_edges(self.graph.ISSUE_PR_HIST):
            resolved.add(edge["source_id"])
        return resolved

    def _diversify(self, scored: list[dict], top_k: int) -> list[dict]:
        if len(scored) <= top_k:
            return scored

        selected = [scored[0]]
        remaining = scored[1:]

        for _ in range(top_k - 1):
            if not remaining:
                break
            best = None
            best_min_sim = -1.0
            for i, cand in enumerate(remaining):
                cid = cand["id"]
                max_sim = 0.0
                for sel in selected:
                    if self.graph.are_issues_connected(cid, sel["id"]):
                        edge = self.graph.get_edge_weight(
                            cid, sel["id"], self.graph.ISSUE_ISSUE_SIM
                        )
                        if edge is not None:
                            max_sim = max(max_sim, edge)
                min_sim = 1.0 - max_sim
                if min_sim > best_min_sim:
                    best_min_sim = min_sim
                    best = i
            if best is not None:
                selected.append(remaining.pop(best))
            else:
                selected.append(remaining.pop(0))

        return selected

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
