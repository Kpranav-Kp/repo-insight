import logging
import math
from datetime import datetime

from .embeddings import EmbeddingService, SearchResult

logger = logging.getLogger(__name__)


class NodeStore:
    def __init__(self, embedding_service: EmbeddingService):
        self._service = embedding_service
        self.meta: list[dict] = []
        self._id_to_idx: dict[str, int] = {}

    def add(self, text: str, metadata: dict) -> int:
        node_id = str(metadata.get("id", ""))
        idx = len(self.meta)
        metadata["_text"] = text
        self.meta.append(metadata)
        if node_id:
            self._id_to_idx[node_id] = idx
        return idx

    def build_index(self):
        items = [
            (m.get("id", str(i)), m.get("_text", ""), m)
            for i, m in enumerate(self.meta)
        ]
        self._service.build_index(items)

    def search(self, query_text: str, top_k: int = 5) -> list[dict]:
        results: list[SearchResult] = self._service.search(query_text, top_k=top_k)
        output = []
        for r in results:
            entry = {**r.metadata, "score": round(r.score, 4)}
            entry.pop("_text", None)
            output.append(entry)
        return output

    def get_vector(self, idx: int):
        text = self.meta[idx].get("_text", "")
        return self._service.encode([text])[0]

    def get_idx_by_id(self, node_id: str) -> int | None:
        return self._id_to_idx.get(str(node_id))

    def __len__(self):
        return len(self.meta)


class AdjacencyTable:
    def __init__(self):
        self.edges: list[dict] = []

    def add_edge(
        self,
        source_type: str,
        source_id: str,
        target_type: str,
        target_id: str,
        relation: str,
        weight: float,
    ):
        self.edges.append(
            {
                "source_type": source_type,
                "source_id": source_id,
                "target_type": target_type,
                "target_id": target_id,
                "relation": relation,
                "weight": round(weight, 4),
            }
        )

    def get_edges(self, relation: str | None = None) -> list[dict]:
        if relation:
            return [e for e in self.edges if e["relation"] == relation]
        return self.edges


class SemanticGraph:
    SKILL_ISSUE_SIM = "SKILL_ISSUE_SIM"
    ISSUE_ISSUE_SIM = "ISSUE_ISSUE_SIM"
    ISSUE_PR_HIST = "ISSUE_PR_HIST"

    DEDUP_THRESHOLD = 0.90
    SKILL_ISSUE_THRESHOLD = 0.25
    ISSUE_ISSUE_THRESHOLD = 0.50

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.skills = NodeStore(
            embedding_service=EmbeddingService(model_name=model_name)
        )
        self.issues = NodeStore(
            embedding_service=EmbeddingService(model_name=model_name)
        )
        self.prs = NodeStore(embedding_service=EmbeddingService(model_name=model_name))
        self.adj = AdjacencyTable()

    def add_skill(self, skill_name: str):
        existing_names = {m["name"] for m in self.skills.meta}
        if skill_name not in existing_names:
            self.skills.add(
                text=skill_name,
                metadata={"id": skill_name, "name": skill_name},
            )

    def add_issue(self, issue: dict):
        if "id" not in issue or "title" not in issue:
            raise ValueError("Issue dict must contain 'id' and 'title'.")

        text = f"{issue['title']}. {issue.get('summary', '')}"
        self.issues.add(text=text, metadata=issue)

        for skill in issue.get("skills", []):
            self.add_skill(skill)

    def add_pr(self, pr: dict):
        if "id" not in pr or "title" not in pr:
            raise ValueError("PR dict must contain 'id' and 'title'.")
        self.prs.add(text=pr["title"], metadata=pr)

    def build_edges(self):
        self.skills.build_index()
        self.issues.build_index()
        self.prs.build_index()

        self._build_skill_issue_edges()
        self._build_issue_issue_edges()
        self._build_issue_pr_edges()

        si = len(self.adj.get_edges(self.SKILL_ISSUE_SIM))
        ii = len(self.adj.get_edges(self.ISSUE_ISSUE_SIM))
        ip = len(self.adj.get_edges(self.ISSUE_PR_HIST))
        logger.info(
            "Graph edges built — SKILL->ISSUE: %d | ISSUE->ISSUE: %d | ISSUE->PR: %d",
            si,
            ii,
            ip,
        )

    def _build_skill_issue_edges(self):
        if not self.skills.meta or not self.issues.meta:
            return

        skill_texts = [m.get("_text", "") for m in self.skills.meta]
        issue_texts = [m.get("_text", "") for m in self.issues.meta]

        skill_vecs = self.skills._service.encode(skill_texts)
        issue_vecs = self.issues._service.encode(issue_texts)

        sim_matrix = skill_vecs @ issue_vecs.T

        for s_idx, skill_meta in enumerate(self.skills.meta):
            for i_idx, issue_meta in enumerate(self.issues.meta):
                sim = float(sim_matrix[s_idx, i_idx])
                if sim >= self.SKILL_ISSUE_THRESHOLD:
                    self.adj.add_edge(
                        source_type="skill",
                        source_id=skill_meta["name"],
                        target_type="issue",
                        target_id=issue_meta["id"],
                        relation=self.SKILL_ISSUE_SIM,
                        weight=sim,
                    )

    def _build_issue_issue_edges(self):
        if len(self.issues.meta) < 2:
            return

        issue_texts = [m.get("_text", "") for m in self.issues.meta]
        issue_vecs = self.issues._service.encode(issue_texts)
        sim_matrix = issue_vecs @ issue_vecs.T

        n = len(self.issues.meta)
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim >= self.ISSUE_ISSUE_THRESHOLD:
                    self.adj.add_edge(
                        source_type="issue",
                        source_id=self.issues.meta[i]["id"],
                        target_type="issue",
                        target_id=self.issues.meta[j]["id"],
                        relation=self.ISSUE_ISSUE_SIM,
                        weight=sim,
                    )

    def _build_issue_pr_edges(self):
        for pr_meta in self.prs.meta:
            issue_id = pr_meta.get("issue_id")
            if issue_id:
                self.adj.add_edge(
                    source_type="issue",
                    source_id=str(issue_id),
                    target_type="pr",
                    target_id=pr_meta["id"],
                    relation=self.ISSUE_PR_HIST,
                    weight=1.0,
                )

    def skill_to_issue(
        self,
        user_skills: list[str],
        top_k: int = 5,
        band_weights: dict[str, float] | None = None,
    ) -> list[dict]:
        if band_weights is None:
            band_weights = {}
        edges = self.adj.get_edges(self.SKILL_ISSUE_SIM)

        skill_lower = {s.lower() for s in user_skills}
        issue_scores: dict[str, float] = {}
        issue_skills: dict[str, set[str]] = {}

        for e in edges:
            skill_name = e["source_id"].lower()
            if skill_name not in skill_lower:
                continue
            issue_id = e["target_id"]
            bw = band_weights.get(skill_name, 0.3)
            contribution = e["weight"] * bw
            issue_scores[issue_id] = issue_scores.get(issue_id, 0.0) + contribution
            if issue_id not in issue_skills:
                issue_skills[issue_id] = set()
            issue_skills[issue_id].add(skill_name)

        if not issue_scores:
            return []

        sorted_ids = sorted(issue_scores, key=issue_scores.__getitem__, reverse=True)
        top_ids = sorted_ids[:top_k]

        results = []
        for issue_id in top_ids:
            meta = self.get_issue_by_id(issue_id)
            if meta:
                entry = {
                    **meta,
                    "score": round(issue_scores[issue_id], 4),
                    "matched_skills": sorted(issue_skills.get(issue_id, [])),
                }
                results.append(entry)
        return results

    def get_issue_by_id(self, issue_id: str) -> dict | None:
        idx = self.issues.get_idx_by_id(issue_id)
        if idx is None:
            return None
        meta = {k: v for k, v in self.issues.meta[idx].items() if k != "_text"}
        return meta

    def get_connected_issues(self, issue_id: str) -> list[str]:
        connected = []
        for edge in self.adj.get_edges(self.ISSUE_ISSUE_SIM):
            if edge["source_id"] == str(issue_id):
                connected.append(edge["target_id"])
            elif edge["target_id"] == str(issue_id):
                connected.append(edge["source_id"])
        return connected

    def get_issues_by_ids(self, issue_ids: set[str]) -> list[dict]:
        results = []
        for m in self.issues.meta:
            if m.get("id") in issue_ids:
                entry = {k: v for k, v in m.items() if k != "_text"}
                results.append(entry)
        return results

    def are_issues_connected(self, id_a: str, id_b: str) -> bool:
        for edge in self.adj.get_edges(self.ISSUE_ISSUE_SIM):
            if (edge["source_id"] == id_a and edge["target_id"] == id_b) or (
                edge["source_id"] == id_b and edge["target_id"] == id_a
            ):
                return True
        return False

    def get_edge_weight(self, id_a: str, id_b: str, relation: str) -> float | None:
        for edge in self.adj.get_edges(relation):
            source_matches = edge["source_id"] == id_a and edge["target_id"] == id_b
            target_matches = edge["source_id"] == id_b and edge["target_id"] == id_a
            if source_matches or target_matches:
                return edge["weight"]
        return None

    def novelty_score(self, recommendation_text: str, issue_id: str) -> float:
        pr_edges = [
            e
            for e in self.adj.get_edges(self.ISSUE_PR_HIST)
            if e["source_id"] == str(issue_id)
        ]
        if not pr_edges:
            return 1.0

        pr_ids = {e["target_id"] for e in pr_edges}
        rec_vec = self.issues._service.encode([recommendation_text])[0]

        max_weighted_sim = 0.0
        now = datetime.now()
        for pr_meta in self.prs.meta:
            if pr_meta["id"] not in pr_ids:
                continue

            pr_idx = self.prs.get_idx_by_id(pr_meta["id"])
            if pr_idx is None:
                continue

            pr_vec = self.prs.get_vector(pr_idx)
            sim = float(rec_vec @ pr_vec)
            created_at_str = pr_meta.get("created_at")
            if created_at_str:
                try:
                    if isinstance(created_at_str, str):
                        normalized = created_at_str.replace("Z", "+00:00")
                        created_at = datetime.fromisoformat(normalized)
                    elif isinstance(created_at_str, datetime):
                        created_at = created_at_str
                    else:
                        raise TypeError(
                            f"Unexpected type for created_at: {type(created_at_str)}"
                        )
                    months = (now - created_at).days / 30.44
                    decay = math.exp(-0.5 * months)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        "Failed to parse created_at '%s': %s", created_at_str, e
                    )
                    decay = 1.0
            else:
                decay = 1.0

            weighted_sim = sim * decay
            if weighted_sim > max_weighted_sim:
                max_weighted_sim = weighted_sim

        return round(1.0 - max_weighted_sim, 4)

    def get_issue_difficulty_score(self, issue_id: str) -> float:
        idx = self.issues.get_idx_by_id(issue_id)
        if idx is None:
            return 0.5
        difficulty = self.issues.meta[idx].get("difficulty", "intermediate")
        return (
            1.0
            if difficulty == "beginner"
            else (0.0 if difficulty == "advanced" else 0.5)
        )

    def get_issue_label_bonus(self, issue_id: str) -> float:
        idx = self.issues.get_idx_by_id(issue_id)
        if idx is None:
            return 0.0
        labels = self.issues.meta[idx].get("labels", [])
        label_lower = [label.lower() for label in labels]
        bonus = 0.0
        if "good first issue" in label_lower:
            bonus += 0.2
        if "help wanted" in label_lower:
            bonus += 0.15
        if "documentation" in label_lower:
            bonus += 0.1
        return min(bonus, 0.3)

    def stats(self) -> dict:
        return {
            "skills": len(self.skills),
            "issues": len(self.issues),
            "prs": len(self.prs),
            "edges": len(self.adj.edges),
        }
