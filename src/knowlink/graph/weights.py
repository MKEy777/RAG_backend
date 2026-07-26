from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..settings import settings


@dataclass
class Entity:
    entity_id: str
    name: str
    description: str = ""
    type: str = ""
    source_chunk_ids: list[str] = field(default_factory=list)
    frequency: int = 1
    in_degree: int = 0
    out_degree: int = 0

    @property
    def degree(self) -> int:
        return self.in_degree + self.out_degree


@dataclass
class Relation:
    source: str
    target: str
    description: str = ""
    strength: float = 5.0
    weight: float = 1.0
    rel_freq: int = 1
    source_chunk_ids: list[str] = field(default_factory=list)


def compute_pmi_weights(
    entities: dict[str, Entity], relations: list[Relation]
) -> list[Relation]:
    if not relations:
        return relations

    total_entity_occ = sum(e.frequency for e in entities.values())
    total_rel_occ = sum(r.rel_freq for r in relations)
    if total_entity_occ == 0 or total_rel_occ == 0:
        return relations

    for rel in relations:
        src_ent = entities.get(rel.source)
        tgt_ent = entities.get(rel.target)
        if not src_ent or not tgt_ent:
            rel.weight = 1.0
            continue

        p_source = src_ent.frequency / total_entity_occ
        p_target = tgt_ent.frequency / total_entity_occ
        p_joint = rel.rel_freq / total_rel_occ

        denom = p_source * p_target
        if denom > 0:
            pmi = max(math.log2(p_joint / denom), 0.0)
        else:
            pmi = 0.0
        rel._pmi = pmi  # type: ignore

    max_pmi = max((getattr(r, "_pmi", 0.0) for r in relations), default=1.0) or 1.0
    max_strength = max((r.strength for r in relations), default=1.0) or 1.0

    for rel in relations:
        pmi = getattr(rel, "_pmi", 0.0)
        norm_pmi = pmi / max_pmi
        norm_strength = rel.strength / max_strength
        combined = norm_pmi * settings.graph_pmi_weight + norm_strength * settings.graph_strength_weight
        rel.weight = 1.0 + 9.0 * combined

    return relations


def compute_degrees(entities: dict[str, Entity], relations: list[Relation]) -> None:
    for ent in entities.values():
        ent.in_degree = 0
        ent.out_degree = 0
    for rel in relations:
        src = entities.get(rel.source)
        tgt = entities.get(rel.target)
        if src:
            src.out_degree += 1
        if tgt:
            tgt.in_degree += 1
