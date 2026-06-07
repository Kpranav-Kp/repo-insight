# backend/core/services/learning_path.py
from collections import defaultdict, deque


class SkillDependencyGraph:
    def __init__(self):
        self.skills: set[str] = set()
        self.edges: dict[str, set[str]] = defaultdict(set)  # skill -> depends_on
        self.issue_skills: list[set[str]] = []  # skills per issue

    def add_issue(self, issue_skills: list[str]) -> None:
        """Add an issue's skills to the graph."""
        skills_set = set(s.lower() for s in issue_skills if s)
        if skills_set:
            self.issue_skills.append(skills_set)
            self.skills.update(skills_set)

    def build_dependencies(self) -> None:
        """
        Build edges based on co-occurrence in issues.
        If skill A and B frequently appear together, assume B depends on A.
        We'll infer simpler skills are prerequisites for more complex ones.
        """
        if not self.issue_skills:
            return

        # Count co-occurrences
        co_occurrence = defaultdict(lambda: defaultdict(int))
        for skills in self.issue_skills:
            skills_list = sorted(list(skills))
            for i, s1 in enumerate(skills_list):
                for s2 in skills_list[i + 1 :]:
                    co_occurrence[s1][s2] += 1
                    co_occurrence[s2][s1] += 1

        # Build edges: simpler skill -> more complex skill
        # Heuristic: if skill A appears in fewer issues, it's likely prerequisite for B
        skill_frequency = defaultdict(int)
        for skills in self.issue_skills:
            for skill in skills:
                skill_frequency[skill] += 1

        for skills in self.issue_skills:
            if len(skills) > 1:
                sorted_skills = sorted(list(skills), key=lambda x: skill_frequency[x])
                # Add edges from less frequent (prerequisite) to more frequent
                for i in range(len(sorted_skills) - 1):
                    self.edges[sorted_skills[i]].add(sorted_skills[i + 1])

    def topological_sort(self) -> list[str]:
        """
        Kahn's algorithm for topological sort.
        Returns ordered list of skills (prerequisites first).
        """
        in_degree = {skill: 0 for skill in self.skills}
        for skill in self.edges:
            for neighbor in self.edges[skill]:
                in_degree[neighbor] += 1

        queue = deque([skill for skill in self.skills if in_degree[skill] == 0])
        result = []

        while queue:
            skill = queue.popleft()
            result.append(skill)

            for neighbor in self.edges[skill]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If cycle exists, include remaining skills
        if len(result) < len(self.skills):
            result.extend([s for s in self.skills if s not in result])

        return result

    def get_learning_path(self, user_skills: set[str]) -> list[str]:
        """
        Return ordered list of skills the user should learn next.
        Filters out already known skills.
        """
        user_skills_lower = set(s.lower() for s in user_skills)
        self.build_dependencies()
        sorted_skills = self.topological_sort()
        return [s for s in sorted_skills if s not in user_skills_lower]


def generate_learning_path(repo_issues: list[dict]) -> str:
    graph = SkillDependencyGraph()

    for issue in repo_issues:
        skills = issue.get("skills", [])
        graph.add_issue(skills)

    ordered_skills = graph.topological_sort()

    if not ordered_skills:
        return (
            "No learning path available. The repository has no clearly defined skills."
        )

    skill_list = "\n".join(
        f"{i + 1}. **{skill}**" for i, skill in enumerate(ordered_skills[:10])
    )

    return (
        f"Currently, there are no issues that match your skillset. "
        f"However, if you plan to contribute to this repository, "
        f"here's a suggested learning path:\n\n{skill_list}\n\n"
        f"Learn these skills in order, and come back once you're familiar with them!"
    )
