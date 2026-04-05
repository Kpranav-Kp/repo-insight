import logging
from typing import Optional
from .embeddings import EmbeddingService, SearchResult

logger = logging.getLogger(__name__)



class NodeStore:
    """
    Wraps EmbeddingService for one node type.
    Stores metadata and provides an id→index lookup
    to avoid O(n) .index() scans.
    """

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
        """Call once after all nodes are added."""
        items = [
            (m.get("id", str(i)), m.get("_text", ""), m)
            for i, m in enumerate(self.meta)
        ]
        self._service.build_index(items)

    def search(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Return top-K metadata dicts with an added 'score' key."""
        results: list[SearchResult] = self._service.search(query_text, top_k=top_k)
        output = []
        for r in results:
            entry = {**r.metadata, "score": round(r.score, 4)}
            entry.pop("_text", None)  
            output.append(entry)
        return output

    def get_vector(self, idx: int):
        """Return the embedding vector for a node by its list index."""
        node_id = self.meta[idx].get("id", str(idx))
        text    = self.meta[idx].get("_text", "")
        return self._service.encode([text])[0]

    def get_idx_by_id(self, node_id: str) -> Optional[int]:
        """O(1) lookup of list index by node id string."""
        return self._id_to_idx.get(str(node_id))

    def __len__(self):
        return len(self.meta)



class AdjacencyTable:
    """
    In-memory edge store.
    Production: migrate this to a PostgreSQL table with Django ORM.
    Each edge: source_type, source_id, target_type, target_id, relation, weight
    """

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
        self.edges.append({
            "source_type": source_type,
            "source_id":   source_id,
            "target_type": target_type,
            "target_id":   target_id,
            "relation":    relation,
            "weight":      round(weight, 4),
        })

    def get_edges(self, relation: Optional[str] = None) -> list[dict]:
        if relation:
            return [e for e in self.edges if e["relation"] == relation]
        return self.edges



class SemanticGraph:
    """
    Full semantic graph.

    Usage
    -----
    sg = SemanticGraph()
    sg.add_issue({"id": "101", "title": "Fix login bug",
                  "summary": "...", "skills": ["python", "flask"]})
    sg.add_pr({"id": "pr1", "title": "Fixed via ORM", "issue_id": "101"})
    sg.build_edges()

    matches = sg.skill_to_issue(["python", "sql"])
    is_dup, match = sg.is_duplicate_issue("SQL injection in auth")
    novelty = sg.novelty_score("Fix using parameterised queries", issue_id="101")
    """

    SKILL_ISSUE_SIM = "SKILL_ISSUE_SIM"
    ISSUE_ISSUE_SIM = "ISSUE_ISSUE_SIM"
    ISSUE_PR_HIST   = "ISSUE_PR_HIST"

    DEDUP_THRESHOLD       = 0.90
    SKILL_ISSUE_THRESHOLD = 0.35  
    ISSUE_ISSUE_THRESHOLD = 0.50

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        shared_service = EmbeddingService(model_name=model_name)
        
        self.skills = NodeStore(embedding_service=shared_service)
        self.issues = NodeStore(embedding_service=shared_service)
        self.prs    = NodeStore(embedding_service=shared_service)
        self.adj    = AdjacencyTable()


    def add_skill(self, skill_name: str):
        """Add a skill node. Skips if already present."""
        existing_names = {m["name"] for m in self.skills.meta}
        if skill_name not in existing_names:
            self.skills.add(
                text=skill_name,
                metadata={"id": skill_name, "name": skill_name},
            )

    def add_issue(self, issue: dict):
        """
        Add an issue node.
        Required keys: id, title
        Optional keys: summary, skills (list), labels, state
        """
        if "id" not in issue or "title" not in issue:
            raise ValueError("Issue dict must contain 'id' and 'title'.")

        text = f"{issue['title']}. {issue.get('summary', '')}"
        self.issues.add(text=text, metadata=issue)

        for skill in issue.get("skills", []):
            self.add_skill(skill)

    def add_pr(self, pr: dict):
        """
        Add a PR node.
        Required keys: id, title
        Optional keys: issue_id (which issue this PR closes)
        """
        if "id" not in pr or "title" not in pr:
            raise ValueError("PR dict must contain 'id' and 'title'.")
        self.prs.add(text=pr["title"], metadata=pr)


    def build_edges(self):
        """
        Compute all edge types.
        Call once after all nodes are added.
        """
        self.skills.build_index()
        self.issues.build_index()
        self.prs.build_index()
        
        self._build_skill_issue_edges()
        self._build_issue_issue_edges()
        self._build_issue_pr_edges()
        logger.info(
            "Graph edges built — SKILL→ISSUE: %d | ISSUE→ISSUE: %d | ISSUE→PR: %d",
            len(self.adj.get_edges(self.SKILL_ISSUE_SIM)),
            len(self.adj.get_edges(self.ISSUE_ISSUE_SIM)),
            len(self.adj.get_edges(self.ISSUE_PR_HIST)),
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
                    source_type = "issue",
                    source_id   = str(issue_id),
                    target_type = "pr",
                    target_id   = pr_meta["id"],
                    relation    = self.ISSUE_PR_HIST,
                    weight      = 1.0,
                )


    def skill_to_issue(self, user_skills: list[str], top_k: int = 5) -> list[dict]:
        """
        Given user skills, return top-K matching issues.
        Uses FAISS search on the issue NodeStore.
        """
        query_text = " ".join(user_skills)
        results = self.issues.search(query_text, top_k=top_k)
        return [r for r in results if r["score"] > 0]


    def is_duplicate_issue(self, new_issue_text: str) -> tuple[bool, Optional[dict]]:
        """
        Returns (True, matched_issue) if similarity >= DEDUP_THRESHOLD,
        else (False, None).
        """
        results = self.issues.search(new_issue_text, top_k=1)
        if results and results[0]["score"] >= self.DEDUP_THRESHOLD:
            return True, results[0]
        return False, None


    def novelty_score(self, recommendation_text: str, issue_id: str) -> float:
        """
        novelty = 1 - max_cosine_sim(recommendation, existing PRs for this issue)
        1.0 = fully novel  |  0.0 = already done

        Uses O(1) id lookup via NodeStore._id_to_idx — no list scanning.
        """
        pr_edges = [
            e for e in self.adj.get_edges(self.ISSUE_PR_HIST)
            if e["source_id"] == str(issue_id)
        ]
        if not pr_edges:
            return 1.0

        pr_ids = {e["target_id"] for e in pr_edges}
        rec_vec = self.issues._service.encode([recommendation_text])[0]

        max_sim = 0.0
        for pr_meta in self.prs.meta:
            if pr_meta["id"] not in pr_ids:
                continue

            pr_idx = self.prs.get_idx_by_id(pr_meta["id"])
            if pr_idx is None:
                continue

            pr_vec = self.prs.get_vector(pr_idx)
            sim    = float(rec_vec @ pr_vec)
            if sim > max_sim:
                max_sim = sim

        return round(1.0 - max_sim, 4)


    def stats(self) -> dict:
        return {
            "skills": len(self.skills),
            "issues": len(self.issues),
            "prs":    len(self.prs),
            "edges":  len(self.adj.edges),
        }