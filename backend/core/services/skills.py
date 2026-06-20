from __future__ import annotations

import re

import numpy as np
from sentence_transformers import SentenceTransformer


class SkillExtractor:
    # Universal concepts only — specific languages / frameworks / tools
    # are discovered dynamically from each repo (Languages API, Topics API,
    # dependency files). This seed exists to catch cross-cutting concepts
    # that don't correspond to any single technology.
    SKILLS_DB = {
        "api",
        "authentication",
        "authorization",
        "caching",
        "ci/cd",
        "cli",
        "configuration",
        "database",
        "deployment",
        "documentation",
        "error handling",
        "logging",
        "migration",
        "monitoring",
        "networking",
        "observability",
        "performance",
        "refactoring",
        "security",
        "serialization",
        "serverless",
        "storage",
        "testing",
        "validation",
        "webhook",
    }

    MATCH_THRESHOLD = 0.35

    def __init__(self, model=None, custom_skills: list[str] | None = None):
        self._model: SentenceTransformer | None = model
        self.skills = self.SKILLS_DB.copy()
        if custom_skills:
            self.skills.update(s.lower() for s in custom_skills)
        self._skill_list = sorted(self.skills)
        self._skill_embeddings: np.ndarray | None = None

    def _get_skill_embeddings(self) -> np.ndarray | None:
        if self._skill_embeddings is not None:
            return self._skill_embeddings
        if self._model is None:
            return None
        self._skill_embeddings = np.asarray(
            self._model.encode(
                self._skill_list,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        ).astype("float32")
        return self._skill_embeddings

    def extract(self, text: str) -> list[str]:
        if not text:
            return []
        emb = self._get_skill_embeddings()
        model = self._model
        if emb is not None and model is not None:
            text_vec = np.asarray(
                model.encode([text], normalize_embeddings=True, show_progress_bar=False)
            ).astype("float32")
            sims = text_vec @ emb.T
            mask = sims[0] > self.MATCH_THRESHOLD
            found = {
                self._skill_list[i] for i in range(len(self._skill_list)) if mask[i]
            }
        else:
            found = set()
            for skill in self._skill_list:
                pattern = re.compile(
                    rf"(?:^|\s)({re.escape(skill)})(?=\s|$)", re.IGNORECASE
                )
                if pattern.search(text):
                    found.add(skill)

        return sorted(found)

    def add_skills(self, skills: list[str]):
        new = [s.lower() for s in skills if s.lower() not in self.skills]
        if not new:
            return
        self.skills.update(new)
        self._skill_list = sorted(self.skills)
        self._skill_embeddings = None


def extract_issue_metadata(
    title: str, body: str, labels: list[str], extractor: SkillExtractor | None = None
) -> dict:
    if extractor is None:
        extractor = SkillExtractor()
    text = f"{title} {body}"
    skills = extractor.extract(text)

    difficulty = "intermediate"

    label_lower = [label.lower() for label in labels]
    beginner_labels = {"good first issue", "beginner", "easy", "help wanted"}
    beginner_keywords = {
        "typo",
        "documentation",
        "docs",
        "example",
        "sample",
        "test",
        "readme",
    }

    if any(label in label_lower for label in beginner_labels) or any(
        kw in text.lower() for kw in beginner_keywords
    ):
        difficulty = "beginner"

    advanced_labels = {"breaking change", "core", "performance", "advanced", "compiler"}
    advanced_keywords = {"architecture", "runtime", "compiler", "refactor", "memory"}

    if any(label in label_lower for label in advanced_labels) or any(
        kw in text.lower() for kw in advanced_keywords
    ):
        difficulty = "advanced"

    return {
        "skills": skills,
        "difficulty": difficulty,
    }
