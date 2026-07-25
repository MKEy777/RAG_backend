from __future__ import annotations

import math

from pyrag.graph.weights import Entity, Relation, compute_degrees, compute_pmi_weights


def test_pmi_basic():
    entities = {
        "A": Entity(entity_id="1", name="A", frequency=5, source_chunk_ids=["c1", "c2", "c3", "c4", "c5"]),
        "B": Entity(entity_id="2", name="B", frequency=3, source_chunk_ids=["c1", "c2", "c3"]),
    }
    relations = [
        Relation(source="A", target="B", strength=8, rel_freq=3, source_chunk_ids=["c1", "c2", "c3"]),
    ]
    result = compute_pmi_weights(entities, relations)
    assert len(result) == 1
    assert 1.0 <= result[0].weight <= 10.0


def test_pmi_single_relation():
    entities = {
        "X": Entity(entity_id="1", name="X", frequency=2, source_chunk_ids=["c1", "c2"]),
        "Y": Entity(entity_id="2", name="Y", frequency=2, source_chunk_ids=["c1", "c2"]),
    }
    relations = [
        Relation(source="X", target="Y", strength=10, rel_freq=2, source_chunk_ids=["c1", "c2"]),
    ]
    result = compute_pmi_weights(entities, relations)
    assert result[0].weight == 10.0


def test_degrees():
    entities = {
        "A": Entity(entity_id="1", name="A", frequency=1),
        "B": Entity(entity_id="2", name="B", frequency=1),
        "C": Entity(entity_id="3", name="C", frequency=1),
    }
    relations = [
        Relation(source="A", target="B", strength=5, rel_freq=1),
        Relation(source="A", target="C", strength=5, rel_freq=1),
    ]
    compute_degrees(entities, relations)
    assert entities["A"].out_degree == 2
    assert entities["A"].in_degree == 0
    assert entities["B"].in_degree == 1
    assert entities["C"].in_degree == 1
    assert entities["A"].degree == 2
