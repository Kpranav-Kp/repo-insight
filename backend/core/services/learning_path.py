from collections import defaultdict, deque


def generate_learning_path(graph, user_skills: list[str]) -> str:
    all_skills_in_graph = {m["name"].lower() for m in graph.skills.meta}
    user_skills_lower = set(s.lower() for s in user_skills)
    unknown = all_skills_in_graph - user_skills_lower

    if not unknown:
        return "You already know all the skills aligned with this repository's open issues."

    prereq_edges = graph.adj.get_edges(graph.SKILL_PREREQ)

    in_degree: dict[str, int] = defaultdict(int)
    dependents: dict[str, list[str]] = defaultdict(list)

    for e in prereq_edges:
        src = e["source_id"].lower()
        tgt = e["target_id"].lower()
        if src in unknown and tgt in unknown:
            dependents[src].append(tgt)
            in_degree[tgt] += 1
            in_degree.setdefault(src, 0)

    for skill in unknown:
        in_degree.setdefault(skill, 0)

    queue: deque = deque(s for s in unknown if in_degree[s] == 0 and s in dependents)
    ordered: list[str] = []

    while queue:
        skill = queue.popleft()
        ordered.append(skill)
        for dep in dependents.get(skill, []):
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    in_dag = set(ordered)
    remaining = sorted(s for s in unknown if s not in in_dag)

    ordered = ordered + remaining

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
        "\n\nStart with the foundational skills first, "
        "then move to skills that build on them."
    )

    return path_text
