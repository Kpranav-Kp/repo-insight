import json
import math
import os
from collections import Counter
from datetime import datetime
from typing import cast

import numpy as np
from django.conf import settings

from .github import GitHubClient, GitHubError, RateLimitError
from .semantic_graph import SemanticGraph
from .skills import SkillExtractor, extract_issue_metadata_batch

BAND_WEIGHTS = {
    "heard_of": 0.1,
    "beginner": 0.3,
    "intermediate": 0.6,
    "advanced": 1.0,
}

NON_ACTIONABLE_LABELS: set[str] = {
    "duplicate",
    "resolution: duplicate",
    "status: duplicate",
    "wontfix",
    "won't fix",
    "resolution: wontfix",
    "invalid",
    "resolution: invalid",
    "question",
    "by design",
    "stale",
    "do not merge",
    "status: blocked",
    "status: stale",
}


class RecommendationEngine:
    def __init__(self, github_token: str | None = None):
        self.github = GitHubClient(token=github_token)
        model_path = getattr(settings, "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        self.graph = SemanticGraph(model_name=model_path)
        self.skill_extractor = SkillExtractor(model=self.graph.issues._service.model)
        self._is_built = False
        self._repo_language: str | None = None
        self._skill_degree_threshold = getattr(settings, "SKILL_DEGREE_THRESHOLD", 3)

    def _compute_skill_score(self, skill_name: str) -> float:
        degree = self.graph.get_skill_degree(skill_name)
        if degree < self._skill_degree_threshold:
            return 0.0
        return min(1.0, degree / 20)

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
            self.graph.add_skill(skill_name)

        skill_counter: Counter = Counter()

        issue_texts = [f"{issue.title} {issue.body}" for issue in issues]
        all_metadata = extract_issue_metadata_batch(
            issue_texts,
            [issue.labels for issue in issues],
            extractor=self.skill_extractor,
        )

        for issue, metadata in zip(issues, all_metadata, strict=False):
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

        # Pre-compute issue vectors so build_index() skips re-encoding
        self.graph.issues._service._vectors = self.graph.issues._service.encode(
            issue_texts
        )

        self.graph.build_edges()
        self._is_built = True

        most_common = skill_counter.most_common()
        skills_found = [
            skill
            for skill, _ in most_common
            if self.graph.get_skill_degree(skill) >= self._skill_degree_threshold
        ]

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
        user=None,
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

        user_skills_set = set(s.lower() for s in user_skill_names)
        feedback_scores = self._feedback_scores(user, user_skills_set) if user else {}
        # Personalized PageRank: multi-hop relevance from user skills
        ppr_scores = self._personalized_pagerank(user_skills_set)

        resolved_ids = self._resolved_issue_ids()

        filtered = []
        for cand in candidates:
            issue_id = cand["id"]
            if cand.get("state") != "open":
                continue
            if issue_id in exclude_issue_ids:
                continue
            cand_labels = {label.lower() for label in cand.get("labels", [])}
            if cand_labels & NON_ACTIONABLE_LABELS:
                continue
            cand["_has_pr_hist"] = issue_id in resolved_ids
            cand["_ppr_score"] = ppr_scores.get(issue_id, 0.0)
            filtered.append(cand)

        if not filtered:
            return []

        # Replace old get_connected_issues() expansion with PPR-discovered issues
        candidate_ids = {c["id"] for c in filtered}
        ppr_issues = sorted(
            [
                (iid, sc)
                for iid, sc in ppr_scores.items()
                if iid not in candidate_ids and iid not in exclude_issue_ids
            ],
            key=lambda x: x[1],
            reverse=True,
        )[: top_k * 3]

        for issue_id, ppr_score in ppr_issues:
            meta = self.graph.get_issue_by_id(issue_id)
            if not meta or meta.get("state") != "open":
                continue
            meta_labels = {label.lower() for label in meta.get("labels", [])}
            if meta_labels & NON_ACTIONABLE_LABELS:
                continue
            meta["score"] = 0.0
            meta["matched_skills"] = []
            meta["_has_pr_hist"] = issue_id in resolved_ids
            meta["_ppr_score"] = ppr_score
            filtered.append(meta)

        if not filtered:
            return []

        now = datetime.now()

        # First pass: compute raw features, track per-candidate min/max
        scored_data = []
        max_degree = 1
        max_pr_density = 0.0
        max_richness = 0.0

        for cand in filtered:
            issue_id = cand["id"]
            issue_skills = set(s.lower() for s in cand.get("skills", []))
            edge_score = float(cand.get("score", 0))

            overlap_skills = user_skills_set & issue_skills
            if overlap_skills:
                fb_score = sum(feedback_scores.get(s, 0.5) for s in overlap_skills) / len(overlap_skills)
            else:
                fb_score = 0.5  # neutral if no skill overlap
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

            # ---- Engineered features ----

            # 1) Issue degree: how many skills connect to this issue
            skill_edges = [
                e
                for e in self.graph.adj.get_edges(self.graph.SKILL_ISSUE_SIM)
                if e["target_id"] == issue_id
            ]
            degree = len(skill_edges)
            if degree > max_degree:
                max_degree = degree

            # 2) Skill entropy: how focused vs. broad the skill match is
            weights = [e["weight"] for e in skill_edges]
            if weights and len(weights) > 1:
                total_w = sum(weights)
                probs = [w / total_w for w in weights]
                entropy = -sum(p * math.log(p) for p in probs) / math.log(len(weights))
            else:
                entropy = 0.0

            # 3) Time-decayed PR density: recent PR activity = healthy issue
            pr_edges = [
                e
                for e in self.graph.adj.get_edges(self.graph.ISSUE_PR_HIST)
                if e["source_id"] == issue_id
            ]
            pr_density = 0.0
            for e in pr_edges:
                pr_meta_list = [
                    m for m in self.graph.prs.meta if m["id"] == e["target_id"]
                ]
                if pr_meta_list:
                    pr_created = pr_meta_list[0].get("created_at", "")
                    if pr_created:
                        try:
                            dt = datetime.fromisoformat(
                                pr_created.replace("Z", "+00:00")
                            )
                            months = (now - dt).days / 30.44
                            pr_density += math.exp(-0.1 * months)
                        except (ValueError, TypeError):
                            pr_density += 0.5
            if pr_density > max_pr_density:
                max_pr_density = pr_density

            # 4) Label richness: maintainer investment signal
            labels = cand.get("labels", [])
            common_labels = {"bug", "enhancement", "question", "feature", "help wanted"}
            richness = len(
                [label for label in labels if label.lower() not in common_labels]
            )
            if richness > max_richness:
                max_richness = richness

            # 5) Prerequisite score: how many issue skills have prerequisites the user knows
            prereq_score = 0.0
            for issue_skill in issue_skills:
                for e in self.graph.adj.get_edges(self.graph.SKILL_PREREQ):
                    if (
                        e["target_id"] == issue_skill
                        and e["source_id"] in user_skills_set
                    ):
                        prereq_score += e["weight"]

            skill_component = skill_overlap if issue_skills else edge_score
            coverage_component = user_coverage if issue_skills else edge_score * 0.5

            scored_data.append(
                {
                    "cand": cand,
                    "skill_component": skill_component,
                    "coverage_component": coverage_component,
                    "edge_score": edge_score,
                    "difficulty_score": difficulty_score,
                    "normalized_label": normalized_label,
                    "recency_score": recency_score,
                    "entropy": entropy,
                    "degree": degree,
                    "pr_density": pr_density,
                    "richness": richness,
                    "overlap_skills": sorted(overlap_skills),
                    "skill_overlap": skill_overlap if issue_skills else edge_score,
                    "pr_penalty": 0.8 if cand.get("_has_pr_hist") else 1.0,
                    "ppr_score": cand.get("_ppr_score", 0.0),
                    "prereq_score": prereq_score,
                     "feedback_score": fb_score,
                }
            )

        # Second pass: normalize and compute final score
        max_degree = max(max_degree, 1)
        max_pr_density = max(max_pr_density, 1.0)
        max_richness = max(max_richness, 1)

        scored = []
        for data in scored_data:
            d = data
            issue_degree = d["degree"] / max_degree
            pr_density_norm = d["pr_density"] / max_pr_density
            richness_norm = d["richness"] / max_richness

            final_score = (
                0.18 * d["skill_component"]
                + 0.10 * d["edge_score"]
                + 0.10 * d["coverage_component"]
                + 0.10 * d["difficulty_score"]
                + 0.05 * d["normalized_label"]
                + 0.10 * d["recency_score"]
                + 0.13 * d["ppr_score"]
                + 0.03 * d["entropy"]
                + 0.05 * issue_degree
                + 0.03 * pr_density_norm
                + 0.04 * richness_norm
                + 0.05 * d["prereq_score"]
                + 0.04 * d["feedback_score"]
            ) * d["pr_penalty"]

            cand = d["cand"]
            cand["skill_overlap"] = d["overlap_skills"]
            cand["match_score"] = round(d["skill_overlap"], 4)
            cand["edge_score"] = round(d["edge_score"], 4)
            cand["difficulty_score"] = d["difficulty_score"]
            cand["label_bonus"] = round(d["normalized_label"] * 0.3, 4)
            cand["recency_score"] = round(d["recency_score"], 4)
            cand["ppr_score"] = round(d["ppr_score"], 4)
            cand["combined_score"] = round(final_score, 4)

            scored.append(cand)

        scored.sort(key=lambda x: x["combined_score"], reverse=True)

        diverse = self._diversify(scored, top_k)
        return diverse[:top_k]

    def _personalized_pagerank(
        self,
        user_skills_lower: set[str],
        alpha: float = 0.85,
        max_iter: int = 20,
        tol: float = 1e-6,
    ) -> dict[str, float]:
        g = self.graph
        id_to_idx: dict[str, int] = {}

        for meta in g.skills.meta:
            id_to_idx[meta["id"]] = len(id_to_idx)
        for meta in g.issues.meta:
            id_to_idx[meta["id"]] = len(id_to_idx)
        for meta in g.prs.meta:
            id_to_idx[meta["id"]] = len(id_to_idx)

        n = len(id_to_idx)
        if n == 0:
            return {}

        adj = np.zeros((n, n), dtype=np.float32)
        for e in g.adj.edges:
            src = id_to_idx.get(e["source_id"])
            tgt = id_to_idx.get(e["target_id"])
            if src is None or tgt is None:
                continue
            multiplier = g.EDGE_TYPE_WEIGHTS.get(e["relation"], 0.3)
            adj[src, tgt] += e["weight"] * multiplier

        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        adj = adj / row_sums

        seed = np.zeros(n, dtype=np.float32)
        for skill_name in user_skills_lower:
            idx = id_to_idx.get(skill_name)
            if idx is not None:
                seed[idx] = 1.0
        seed_sum = seed.sum()
        if seed_sum == 0:
            return {}
        seed = seed / seed_sum

        v = seed.copy()
        for _ in range(max_iter):
            prev = v
            v = (1.0 - alpha) * seed + alpha * (adj.T @ v)
            if float(np.linalg.norm(v - prev, ord=1)) < tol:
                break

        scores = {}
        max_score = 0.0
        for meta in g.issues.meta:
            issue_id = meta["id"]
            idx = id_to_idx.get(issue_id)
            if idx is not None:
                score = float(v[idx])
                if score > max_score:
                    max_score = score
                scores[issue_id] = score

        if max_score > 0:
            for k in scores:
                scores[k] = round(scores[k] / max_score, 4)

        return scores

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
    def _feedback_scores(self, user, user_skills_set: set[str]) -> dict[str, float]:
        """
        Returns a per-skill score in [0, 1] based on this user's past feedback.
        1.0 = strongly liked issues with this skill before
        0.0 = strongly disliked issues with this skill before
        0.5 = no feedback history for this skill (neutral, doesn't help or hurt)
        """
        from ..models import Recommendation

        if not user_skills_set:
            return {}

        past = Recommendation.objects.filter(
            user=user, feedback__isnull=False
        ).values("skills_matched", "feedback")

        skill_stats: dict[str, list[int]] = {}  # skill -> [thumbs_up_count, total_count]
        for rec in past:
            matched = [s.lower() for s in (rec["skills_matched"] or [])]
            overlap = user_skills_set & set(matched)
            if not overlap:
                continue
            for skill in overlap:
                if skill not in skill_stats:
                    skill_stats[skill] = [0, 0]
                skill_stats[skill][1] += 1
                if rec["feedback"]:
                    skill_stats[skill][0] += 1

        return {
            skill: (up / total if total else 0.5)
            for skill, (up, total) in skill_stats.items()
        }

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
            flat = json.load(f)
            self.graph.adj._edges.clear()
            for entry in flat:
                self.graph.adj._edges[entry["relation"]].append(entry)

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
