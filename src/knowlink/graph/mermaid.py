from __future__ import annotations

from .weights import Entity, Relation

HIGHLIGHT_FREQ_THRESHOLD = 5
BOLD_STRENGTH_THRESHOLD = 7


def generate_mermaid(entities: dict[str, Entity], relations: list[Relation]) -> str:
    if not entities or not relations:
        return ""

    sorted_entities = sorted(entities.values(), key=lambda e: e.frequency, reverse=True)
    sorted_relations = sorted(relations, key=lambda r: r.weight, reverse=True)

    adjacency: dict[str, set[str]] = {name: set() for name in entities}
    for rel in sorted_relations:
        if rel.source in adjacency and rel.target in adjacency:
            adjacency[rel.source].add(rel.target)
            adjacency[rel.target].add(rel.source)

    visited: set[str] = set()
    components: list[list[str]] = []
    for start in [e.name for e in sorted_entities]:
        if start in visited:
            continue
        component: list[str] = []
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    stack.append(neighbor)
        if len(component) > 1:
            components.append(component)

    lines = ["graph TD"]
    for i, comp in enumerate(components):
        lines.append(f"    subgraph G{i}")
        for name in comp:
            ent = entities[name]
            node_id = name.replace(" ", "_").replace("/", "_")
            lines.append(f'        {node_id}["{name}"]')
            if ent.frequency > HIGHLIGHT_FREQ_THRESHOLD:
                lines.append(f"        style {node_id} fill:#ffeb3b")
        lines.append("    end")

    for rel in sorted_relations:
        src_id = rel.source.replace(" ", "_").replace("/", "_")
        tgt_id = rel.target.replace(" ", "_").replace("/", "_")
        label = rel.description[:30] if rel.description else ""
        if rel.strength > BOLD_STRENGTH_THRESHOLD:
            lines.append(f'    {src_id} ==>|"{label}"| {tgt_id}')
        else:
            lines.append(f'    {src_id} -->|"{label}"| {tgt_id}')

    return "\n".join(lines)
