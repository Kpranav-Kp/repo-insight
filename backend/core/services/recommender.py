import json
import math
import os
from collections import Counter
from datetime import datetime
from typing import cast

from django.conf import settings

from .github import GitHubClient, GitHubError, RateLimitError
from .semantic_graph import SemanticGraph
from .skills import SkillExtractor, extract_issue_metadata

BAND_WEIGHTS = {
    "heard_of": 0.1,
    "beginner": 0.3,
    "intermediate": 0.6,
    "advanced": 1.0,
}


class RecommendationEngine:
    def __init__(self, github_token: str | None = None):
        self.github = GitHubClient(token=github_token)
        model_path = getattr(settings, "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        self.graph = SemanticGraph(model_name=model_path)
        self.skill_extractor = SkillExtractor(model=self.graph.issues._service.model)
        self._is_built = False
        self._repo_language: str | None = None

    @staticmethod
    def _is_skill_like(name: str) -> bool:
        nl = name.lower()
        if nl in SkillExtractor.SKILLS_DB:
            return True
        if "@" in nl:
            return False
        if nl.count("-") >= 2:
            return False
        if len(nl) < 2:
            return False
        return True

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

        repo_languages = sorted(languages.keys())
        top_lang = (
            max(languages, key=lambda k: languages[k] or 0) if languages else None
        )
        self._repo_language = top_lang

        try:
            topics = self.github.fetch_topics(repo_url)
        except (RateLimitError, GitHubError):
            topics = []

        try:
            dep_skills = self.github.extract_dependency_skills(repo_url)
        except (RateLimitError, GitHubError):
            dep_skills = []

        repo_skills_lower = {s.lower() for s in repo_languages + topics + dep_skills}
        self.skill_extractor.add_skills(sorted(repo_skills_lower))

        for skill_name in sorted(repo_skills_lower):
            if self._is_skill_like(skill_name):
                self.graph.add_skill(skill_name)

        skill_counter: Counter = Counter()

        for issue in issues:
            metadata = extract_issue_metadata(
                issue.title,
                issue.body,
                issue.labels,
                extractor=self.skill_extractor,
            )
            skills = metadata["skills"]
            difficulty = metadata["difficulty"]

            skill_counter.update(s.lower() for s in skills)

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

        most_common = skill_counter.most_common()
        skills_found = [skill for skill, _ in most_common if self._is_skill_like(skill)]

        return {
            "repository_url": repo_url,
            "issues_indexed": len(issues),
            "prs_indexed": len(prs),
            "skills_found": skills_found,
            "graph_stats": self.graph.stats(),
        }

    def recommend(
        self,
        user_skills: list[str] | list[dict],
        top_k: int = 5,
        exclude_issue_ids: set[str] | None = None,
    ) -> list[dict]:
        if not self._is_built:
            raise RuntimeError("Graph not built. Call build_from_repository() first.")

        if user_skills and isinstance(user_skills[0], dict):
            dict_skills = cast("list[dict]", user_skills)
            user_skill_names = [s["skill"] for s in dict_skills]
            skill_bands = {
                s["skill"].lower(): s.get("band", "intermediate") for s in dict_skills
            }
        else:
            str_skills = cast("list[str]", user_skills) if user_skills else []
            user_skill_names = str_skills
            skill_bands = {}

        exclude_issue_ids = exclude_issue_ids or set()

        band_weights_for_search = {
            skill: BAND_WEIGHTS.get(band, 0.3) for skill, band in skill_bands.items()
        }
        candidates = self.graph.skill_to_issue(
            user_skill_names,
            top_k=top_k * 3,
            band_weights=band_weights_for_search,
        )

        if not candidates:
            candidates = self._label_fallback(
                user_skill_names,
                band_weights_for_search,
                top_k=top_k * 3,
            )

        if not candidates:
            return []

        resolved_ids = self._resolved_issue_ids()

        filtered = []
        for cand in candidates:
            issue_id = cand["id"]
            if cand.get("state") != "open":
                continue
            if issue_id in exclude_issue_ids:
                continue
            cand["_has_pr_hist"] = issue_id in resolved_ids
            filtered.append(cand)

        if not filtered:
            return []

        candidate_ids = {c["id"] for c in filtered}
        expanded_ids = set(candidate_ids)
        for cid in list(candidate_ids):
            connected = self.graph.get_connected_issues(cid)
            for conn in connected:
                if conn not in expanded_ids and conn not in exclude_issue_ids:
                    expanded_ids.add(conn)

        seen_ids = {c["id"] for c in filtered}
        if expanded_ids - seen_ids:
            extra = self.graph.get_issues_by_ids(expanded_ids - seen_ids)
            for e in extra:
                e["_has_pr_hist"] = e.get("id", "") in resolved_ids
            filtered.extend(extra)

        user_skills_set = set(s.lower() for s in user_skill_names)
        now = datetime.now()

        scored = []
        for cand in filtered:
            issue_id = cand["id"]
            issue_skills = set(s.lower() for s in cand.get("skills", []))
            edge_score = float(cand.get("score", 0))

            overlap_skills = user_skills_set & issue_skills

            total_weight = 0.0
            matched_weight = 0.0
            for us in user_skills_set:
                bw = BAND_WEIGHTS.get(skill_bands.get(us, "intermediate"), 0.3)
                total_weight += bw
                if us in issue_skills:
                    matched_weight += bw

            if issue_skills:
                issue_weight = 0.0
                for is_ in issue_skills:
                    bw = BAND_WEIGHTS.get(skill_bands.get(is_, "intermediate"), 0.3)
                    issue_weight += bw if is_ in overlap_skills else 0.0
                total_issue_weight = sum(
                    BAND_WEIGHTS.get(skill_bands.get(is_, "intermediate"), 0.3)
                    for is_ in issue_skills
                )
                skill_overlap = (
                    issue_weight / total_issue_weight if total_issue_weight else 0.0
                )
                user_coverage = matched_weight / total_weight if total_weight else 0.0
            else:
                skill_overlap = 0.0
                user_coverage = 0.0

            difficulty_score = self.graph.get_issue_difficulty_score(issue_id)
            label_bonus = self.graph.get_issue_label_bonus(issue_id)
            normalized_label = label_bonus / 0.3 if label_bonus else 0.0

            created_str = cand.get("created_at", "")
            recency_score = 1.0
            if created_str:
                try:
                    dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                    months = (now - dt).days / 30.44
                    recency_score = max(0.0, math.exp(-0.08 * months))
                except (ValueError, TypeError):
                    pass

            skill_component = skill_overlap if issue_skills else edge_score
            coverage_component = user_coverage if issue_skills else edge_score * 0.5

            pr_penalty = 0.5 if cand.get("_has_pr_hist") else 1.0

            final_score = (
                0.25 * skill_component
                + 0.25 * edge_score
                + 0.15 * coverage_component
                + 0.10 * difficulty_score
                + 0.10 * normalized_label
                + 0.15 * recency_score
            ) * pr_penalty

            cand["skill_overlap"] = sorted(overlap_skills)
            cand["match_score"] = round(
                skill_overlap if issue_skills else edge_score, 4
            )
            cand["edge_score"] = round(edge_score, 4)
            cand["difficulty_score"] = difficulty_score
            cand["label_bonus"] = label_bonus
            cand["recency_score"] = round(recency_score, 4)
            cand["combined_score"] = round(final_score, 4)

            scored.append(cand)

        scored.sort(key=lambda x: x["combined_score"], reverse=True)

        diverse = self._diversify(scored, top_k)
        return diverse[:top_k]

    def _label_fallback(
        self,
        user_skills: list[str],
        band_weights: dict[str, float],
        top_k: int,
    ) -> list[dict]:
        user_skills_lower = {s.lower() for s in user_skills}
        scored = []

        for meta in self.graph.issues.meta:
            if meta.get("state") != "open":
                continue
            issue_skills = set(skill.lower() for skill in meta.get("skills", []))
            issue_labels = set(label.lower() for label in meta.get("labels", []))
            matched = user_skills_lower & (issue_skills | issue_labels)
            if not matched:
                continue

            score = sum(band_weights.get(s, 0.3) for s in matched)
            entry = {k: v for k, v in meta.items() if k != "_text"}
            entry["score"] = round(score, 4)
            entry["matched_skills"] = sorted(matched)
            scored.append(entry)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _resolved_issue_ids(self) -> set[str]:
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

    def save_index(self, directory: str):
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
