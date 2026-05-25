# backend/core/services/skills.py
import re


class SkillExtractor:
    SKILLS_DB = {
        # Languages
        "python",
        "javascript",
        "typescript",
        "java",
        "go",
        "rust",
        "c++",
        "c#",
        "ruby",
        "php",
        # Web
        "react",
        "vue",
        "angular",
        "svelte",
        "next.js",
        "nuxt",
        "html",
        "css",
        "sass",
        "less",
        # Backend
        "django",
        "flask",
        "fastapi",
        "express",
        "spring",
        "rails",
        "laravel",
        # Databases
        "sql",
        "postgresql",
        "mysql",
        "sqlite",
        "mongodb",
        "redis",
        "elasticsearch",
        "dynamodb",
        # DevOps
        "docker",
        "kubernetes",
        "terraform",
        "ansible",
        "jenkins",
        "github actions",
        "gitlab ci",
        # Cloud
        "aws",
        "gcp",
        "azure",
        "vercel",
        "netlify",
        "heroku",
        # ML/AI
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "pandas",
        "numpy",
        "machine learning",
        "ai",
        "llm",
        # Tools
        "git",
        "webpack",
        "vite",
        "babel",
        "eslint",
        "prettier",
        # Concepts
        "api",
        "graphql",
        "rest",
        "websocket",
        "oauth",
        "jwt",
        "authentication",
        "authorization",
        "microservices",
        "serverless",
        "ci/cd",
        "testing",
        "tdd",
        "agile",
    }

    def __init__(self, custom_skills: list[str] | None = None):
        self.skills = self.SKILLS_DB.copy()
        if custom_skills:
            self.skills.update(s.lower() for s in custom_skills)
        self._compile_patterns()

    def _compile_patterns(self):
        self._patterns = {
            skill: re.compile(rf"(?:^|\s)({re.escape(skill)})(?=\s|$)", re.IGNORECASE)
            for skill in self.skills
        }

    def extract(self, text: str) -> list[str]:
        if not text:
            return []

        found = set()
        for skill, pattern in self._patterns.items():
            if pattern.search(text):
                found.add(skill)

        return sorted(found)

    def _format_skill(self, skill: str) -> str:
        return " ".join(word.capitalize() for word in skill.split())

    def add_skills(self, skills: list[str]):
        self.skills.update(s.lower() for s in skills)
        self._compile_patterns()
