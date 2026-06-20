from collections import defaultdict, deque


def _build_adjacency(graph):
    """Build bidirectional skill↔issue and issue↔issue maps from graph edges."""
    skill_to_issues: dict[str, set[str]] = defaultdict(set)
    issue_to_skills: dict[str, set[str]] = defaultdict(set)
    issue_to_issues: dict[str, set[str]] = defaultdict(set)

    for edge in graph.adj.get_edges():
        if edge["relation"] == graph.SKILL_ISSUE_SIM:
            skill_id = edge["source_id"]
            issue_id = edge["target_id"]
            skill_to_issues[skill_id].add(issue_id)
            issue_to_skills[issue_id].add(skill_id)
        elif edge["relation"] == graph.ISSUE_ISSUE_SIM:
            issue_to_issues[edge["source_id"]].add(edge["target_id"])
            issue_to_issues[edge["target_id"]].add(edge["source_id"])

    # Also include skills from issue metadata that may not have embedding edges
    for issue_meta in graph.issues.meta:
        issue_id = issue_meta.get("id", "")
        for skill in issue_meta.get("skills", []):
            skill_lower = skill.lower()
            skill_to_issues[skill_lower].add(issue_id)
            issue_to_skills[issue_id].add(skill_lower)

    return skill_to_issues, issue_to_skills, issue_to_issues


def generate_learning_path(graph, user_skills: list[str]) -> str:
    skill_to_issues, issue_to_skills, issue_to_issues = _build_adjacency(graph)

    all_skills_in_graph = set(skill_to_issues.keys())
    user_skills_lower = set(s.lower() for s in user_skills)
    unknown = all_skills_in_graph - user_skills_lower

    if not unknown:
        return "You already know all the skills aligned with this repository's open issues."

    # BFS from user skills: skill → issue → issue → skill
    visited_skills = set(user_skills_lower)
    visited_issues: set[str] = set()
    queue: deque = deque()
    for s in user_skills_lower:
        if s in skill_to_issues:
            queue.append((s, 0, "skill"))

    distance: dict[str, int] = {}

    while queue:
        node, dist, node_type = queue.popleft()
        if node_type == "skill":
            for issue_id in skill_to_issues.get(node, set()):
                if issue_id not in visited_issues:
                    visited_issues.add(issue_id)
                    queue.append((issue_id, dist + 1, "issue"))
        elif node_type == "issue":
            for skill_name in issue_to_skills.get(node, set()):
                skill_lower = skill_name.lower()
                if skill_lower not in visited_skills:
                    visited_skills.add(skill_lower)
                    if skill_lower not in distance:
                        distance[skill_lower] = dist + 1
                    queue.append((skill_lower, dist + 1, "skill"))
            for rel_id in issue_to_issues.get(node, set()):
                if rel_id not in visited_issues:
                    visited_issues.add(rel_id)
                    queue.append((rel_id, dist + 1, "issue"))

    reachable = [s for s in unknown if s.lower() in distance]
    unreachable = [s for s in unknown if s.lower() not in distance]

    reachable.sort(key=lambda s: (distance[s.lower()], s.lower()))
    unreachable.sort()

    ordered = reachable + unreachable

    if not ordered:
        return (
            "No learning path available. The repository has no clearly defined skills."
        )

    heading = (
        "Based on your current skills, here is a suggested learning path "
        "to help you contribute more effectively to this repository:\n\n"
    )

    skill_lines = []
    for i, skill in enumerate(ordered[:15]):
        skill_lines.append(f"{i + 1}. **{skill}**")

    path_text = heading + "\n".join(skill_lines)

    if len(ordered) > 15:
        path_text += f"\n\n... and {len(ordered) - 15} more skills."

    path_text += (
        "\n\nStart with the skills closest to what you already know, "
        "and gradually expand outward as you contribute."
    )

    return path_text
