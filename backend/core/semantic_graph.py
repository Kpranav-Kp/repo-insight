"""
semantic_graph.py
=================
Semantic Graph for OSS Contribution Pathfinder (Module 4)

Based on project spec:
- Nodes  : User, Skill, Issue, PR
- Edges  : Skill→Issue (similarity), Issue→Issue (similarity), Issue→PR (historical)
- Storage: Embeddings in FAISS + adjacency table in PostgreSQL
- Purpose: Avoid duplicates, diversify output, cluster similar issues
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── MODEL ─────────────────────────────────────────────────────────────────────

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed(text: str) -> np.ndarray:
    """Convert text to normalised embedding vector (384-dim)."""
    return get_model().encode(text, normalize_embeddings=True)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalised vectors."""
    return float(np.dot(a, b))

# ── NODE STORES ───────────────────────────────────────────────────────────────
# Each store is just:
#   vectors : numpy array of shape (N, 384)
#   index   : FAISS index for fast similarity search
#   meta    : list of dicts with node data

class NodeStore:
    """Stores embeddings + metadata for one node type."""

    def __init__(self, dim: int = 384):
        self.index   = faiss.IndexFlatIP(dim)  # Inner Product = cosine sim (on normalised vecs)
        self.vectors = []                        # list of np arrays
        self.meta    = []                        # list of dicts

    def add(self, text: str, metadata: dict):
        vec = embed(text)
        self.vectors.append(vec)
        self.meta.append(metadata)
        self.index.add(np.array([vec]))          # add to FAISS
        return len(self.meta) - 1                # return index position

    def search(self, query_text: str, top_k: int = 5):
        """Return top-K most similar nodes to query_text."""
        query_vec = embed(query_text)
        D, I = self.index.search(np.array([query_vec]), top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            results.append({
                **self.meta[idx],
                "score": round(float(score), 4),
            })
        return results

    def get_vector(self, idx: int) -> np.ndarray:
        return self.vectors[idx]

    def __len__(self):
        return len(self.meta)


# ── ADJACENCY TABLE ───────────────────────────────────────────────────────────
# In production this goes to PostgreSQL.
# For now it's an in-memory list of edges.
# Each edge: { source_type, source_id, target_type, target_id, relation, weight }

class AdjacencyTable:
    def __init__(self):
        self.edges = []

    def add_edge(self, source_type, source_id, target_type, target_id, relation, weight):
        self.edges.append({
            "source_type": source_type,
            "source_id":   source_id,
            "target_type": target_type,
            "target_id":   target_id,
            "relation":    relation,
            "weight":      round(weight, 4),
        })

    def get_edges(self, relation=None):
        if relation:
            return [e for e in self.edges if e["relation"] == relation]
        return self.edges


# ── SEMANTIC GRAPH ────────────────────────────────────────────────────────────

class SemanticGraph:
    """
    The full semantic graph.

    Nodes  : user, skill, issue, pr
    Edges  : skill→issue (SKILL_ISSUE_SIM)
             issue→issue (ISSUE_ISSUE_SIM)
             issue→pr    (ISSUE_PR_HIST)

    Usage
    -----
    sg = SemanticGraph()
    sg.add_user({"id": "u1", "name": "Alice"}, skills=["Python", "SQL"])
    sg.add_issue({"id": "101", "title": "Fix login bug", "summary": "..."})
    sg.add_pr({"id": "pr1", "title": "Fixed login via parameterised queries", "issue_id": "101"})
    sg.build_edges()

    # 3 use cases
    matches = sg.skill_to_issue(["Python", "SQL"])
    is_dup  = sg.is_duplicate_issue("SQL injection in auth endpoint")
    novelty = sg.novelty_score("Fix auth using ORM", issue_id="101")
    """

    SKILL_ISSUE_SIM = "SKILL_ISSUE_SIM"
    ISSUE_ISSUE_SIM = "ISSUE_ISSUE_SIM"
    ISSUE_PR_HIST   = "ISSUE_PR_HIST"

    # thresholds
   
    DEDUP_THRESHOLD       = 0.90
    SKILL_ISSUE_THRESHOLD = 0.20  # was 0.40
    ISSUE_ISSUE_THRESHOLD = 0.50  # was 0.70

    def __init__(self):
        self.users  = NodeStore()
        self.skills = NodeStore()
        self.issues = NodeStore()
        self.prs    = NodeStore()
        self.adj    = AdjacencyTable()

    # ── ADD NODES ─────────────────────────────────────────────────────────────

    def add_user(self, user: dict, skills: list[str]):
        """Add a user node and their skill nodes."""
        self.users.add(user.get("name", "user"), user)
        for skill in skills:
            self.add_skill(skill)

    def add_skill(self, skill_name: str):
        """Add a skill node if not already present."""
        existing = [m["name"] for m in self.skills.meta]
        if skill_name not in existing:
            self.skills.add(skill_name, {"name": skill_name})

    def add_issue(self, issue: dict):
        """
        Add an issue node.
        issue must have: id, title, summary, skills (list of skill names extracted by LLM)
        """
        text = f"{issue['title']}. {issue.get('summary', '')}"
        self.issues.add(text, issue)

        # also make sure all skills from this issue exist as nodes
        for skill in issue.get("skills", []):
            self.add_skill(skill)

    def add_pr(self, pr: dict):
        """
        Add a PR node.
        pr must have: id, title, issue_id (which issue this PR closes)
        """
        self.prs.add(pr["title"], pr)

    # ── BUILD EDGES ───────────────────────────────────────────────────────────

    def build_edges(self):
        """
        Compute all 3 edge types after nodes are added.
        Call this once after adding all issues, skills, PRs.
        """
        self._build_skill_issue_edges()
        self._build_issue_issue_edges()
        self._build_issue_pr_edges()

        print(f"[Graph] Edges built:")
        print(f"  SKILL→ISSUE : {len(self.adj.get_edges(self.SKILL_ISSUE_SIM))}")
        print(f"  ISSUE→ISSUE : {len(self.adj.get_edges(self.ISSUE_ISSUE_SIM))}")
        print(f"  ISSUE→PR    : {len(self.adj.get_edges(self.ISSUE_PR_HIST))}")

    def _build_skill_issue_edges(self):
        """
        Edge: Skill → Issue (similarity)
        For every skill-issue pair, compute cosine sim.
        Add edge if sim >= threshold.
        """
        for s_idx, skill_meta in enumerate(self.skills.meta):
            skill_vec = self.skills.get_vector(s_idx)

            for i_idx, issue_meta in enumerate(self.issues.meta):
                issue_vec = self.issues.get_vector(i_idx)
                sim = cosine_sim(skill_vec, issue_vec)

                if sim >= self.SKILL_ISSUE_THRESHOLD:
                    self.adj.add_edge(
                        source_type = "skill",
                        source_id   = skill_meta["name"],
                        target_type = "issue",
                        target_id   = issue_meta["id"],
                        relation    = self.SKILL_ISSUE_SIM,
                        weight      = sim,
                    )

    def _build_issue_issue_edges(self):
        """
        Edge: Issue → Issue (similarity)
        For every pair of issues, compute cosine sim.
        Add edge if sim >= threshold.
        Used for: duplicate filtering + diversification.
        """
        n = len(self.issues)
        for i in range(n):
            for j in range(i + 1, n):
                sim = cosine_sim(
                    self.issues.get_vector(i),
                    self.issues.get_vector(j),
                )
                if sim >= self.ISSUE_ISSUE_THRESHOLD:
                    self.adj.add_edge(
                        source_type = "issue",
                        source_id   = self.issues.meta[i]["id"],
                        target_type = "issue",
                        target_id   = self.issues.meta[j]["id"],
                        relation    = self.ISSUE_ISSUE_SIM,
                        weight      = sim,
                    )

    def _build_issue_pr_edges(self):
        """
        Edge: Issue → PR (historical relation)
        Links each PR back to the issue it closes.
        Used for: novelty scoring.
        """
        for pr_meta in self.prs.meta:
            issue_id = pr_meta.get("issue_id")
            if issue_id:
                self.adj.add_edge(
                    source_type = "issue",
                    source_id   = issue_id,
                    target_type = "pr",
                    target_id   = pr_meta["id"],
                    relation    = self.ISSUE_PR_HIST,
                    weight      = 1.0,
                )

    # ── USE CASE 1: SKILL → ISSUE MATCHING ───────────────────────────────────
    
    
      # ← add this line
    def skill_to_issue(self, user_skills: list[str], top_k: int = 5) -> list[dict]:
        """
        Given user's skills, return top-K matching issues.
        Called by: Skill Matching Agent
        """
        query_text = " ".join(user_skills)
        results = self.issues.search(query_text, top_k=top_k)
        return [r for r in results if r["score"] > 0]
        
    # ── USE CASE 2: DUPLICATE FILTERING ───────────────────────────────────────

    def is_duplicate_issue(self, new_issue_text: str) -> tuple[bool, dict | None]:
        """
        Check if a new issue is too similar to an existing one.
        Returns (True, matched_issue) or (False, None).
        Called by: GitHub Fetcher (Module 1) before storing new issues.
        """
        results = self.issues.search(new_issue_text, top_k=1)
        if results and results[0]["score"] >= self.DEDUP_THRESHOLD:
            return True, results[0]
        return False, None

    # ── USE CASE 3: NOVELTY SCORING ───────────────────────────────────────────

    def novelty_score(self, recommendation_text: str, issue_id: str) -> float:
        """
        novelty = 1 - max_sim(recommendation, existing PRs for this issue)
        1.0 = completely novel, 0.0 = already done
        Called by: Ranking Engine (Module 6)
        """
        # get all PRs linked to this issue
        pr_edges = [
            e for e in self.adj.get_edges(self.ISSUE_PR_HIST)
            if e["source_id"] == issue_id
        ]
        if not pr_edges:
            return 1.0  # no existing PRs → fully novel

        pr_ids  = [e["target_id"] for e in pr_edges]
        rec_vec = embed(recommendation_text)

        max_sim = 0.0
        for pr_meta in self.prs.meta:
            if pr_meta["id"] in pr_ids:
                pr_idx  = self.prs.meta.index(pr_meta)
                pr_vec  = self.prs.get_vector(pr_idx)
                sim     = cosine_sim(rec_vec, pr_vec)
                max_sim = max(max_sim, sim)

        return round(1.0 - max_sim, 4)

    # ── STATS ─────────────────────────────────────────────────────────────────

    def stats(self):
        return {
            "users":  len(self.users),
            "skills": len(self.skills),
            "issues": len(self.issues),
            "prs":    len(self.prs),
            "edges":  len(self.adj.edges),
        }


# ── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sg = SemanticGraph()

    # add user
    sg.add_user({"id": "u1", "name": "Alice"}, skills=["Python", "SQL"])

    # add issues
    sg.add_issue({"id": "101", "title": "Fix SQL injection in login",
                  "summary": "Auth endpoint vulnerable to SQL injection",
                  "skills": ["Python", "SQL", "Flask"]})

    sg.add_issue({"id": "102", "title": "Add dark mode toggle",
                  "summary": "Users want dark/light theme in navbar",
                  "skills": ["React", "CSS"]})

    sg.add_issue({"id": "103", "title": "Optimise slow DB queries",
                  "summary": "Queries on issues table very slow",
                  "skills": ["Python", "SQL", "PostgreSQL"]})

    # add PRs
    sg.add_pr({"id": "pr1", "title": "Fixed SQL injection using parameterised queries", "issue_id": "101"})
    sg.add_pr({"id": "pr2", "title": "Added CSS dark mode with toggle switch", "issue_id": "102"})

    # build all edges
    sg.build_edges()

    print("\n📊 Stats:", sg.stats())

    print("\n🔍 Issues matching ['Python', 'SQL']:")
    for r in sg.skill_to_issue(["Python", "SQL"]):
        print(f"  [{r['score']}] {r['title']}")

    is_dup, match = sg.is_duplicate_issue("SQL injection vulnerability in authentication")
    print(f"\n🔁 Duplicate check: {is_dup} → {match['title'] if match else None}")

    novelty = sg.novelty_score("Fix SQL injection using ORM instead of raw queries", issue_id="101")
    print(f"\n✨ Novelty score: {novelty}")