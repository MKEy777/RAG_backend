from __future__ import annotations

import math

from knowlink.graph.weights import Entity, Relation, compute_degrees, compute_pmi_weights


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


async def test_relation_chunk_attribution(monkeypatch):
    from knowlink.graph import builder as builder_mod

    entity_by_text = {
        "文本一": '[{"title": "A", "type": "Concept", "description": "a"}]',
        "文本二": '[{"title": "B", "type": "Concept", "description": "b"}]',
        "文本三": "[]",
    }

    class FakeClient:
        async def chat(self, messages, **kw):
            if kw.get("purpose") == "graph:extract_entities":
                return entity_by_text[messages[1]["content"]]
            return '[{"source": "A", "target": "B", "description": "rel", "strength": 8}]'

    monkeypatch.setattr(builder_mod, "get_llm_client", lambda: FakeClient())
    b = builder_mod.GraphBuilder()
    chunks = [
        {"id": "c1", "content": "文本一"},
        {"id": "c2", "content": "文本二"},
        {"id": "c3", "content": "文本三"},
    ]
    ents, rels = await b.build("kb1", "doc1", chunks)
    assert set(ents) == {"A", "B"}
    assert len(rels) == 1
    # 关系归属两实体 chunk 并集，不含无实体的 c3（WeKnora findRelationChunkIDs 语义）
    assert rels[0].source_chunk_ids == ["c1", "c2"]
    assert rels[0].rel_freq == 2


async def test_graph_manager_reconcile(tmp_path):
    from knowlink.graph.builder import GraphManager

    gm = GraphManager(snapshot_root=tmp_path)
    ents = {
        "A": Entity(entity_id="1", name="A", source_chunk_ids=["c1", "dead1"], frequency=2),
        "B": Entity(entity_id="2", name="B", source_chunk_ids=["dead2"], frequency=1),
        "C": Entity(entity_id="3", name="C", source_chunk_ids=["c2"], frequency=1),
    }
    rels = [
        Relation(source="A", target="B", source_chunk_ids=["c1", "dead2"], rel_freq=2),
        Relation(source="A", target="C", source_chunk_ids=["c1", "c2"], rel_freq=2),
    ]
    gm._graphs["kb1"] = (ents, rels)
    removed = await gm.reconcile("kb1", {"c1", "c2"})
    assert removed == 1
    ents2, rels2 = gm._graphs["kb1"]
    assert set(ents2) == {"A", "C"}
    assert ents2["A"].source_chunk_ids == ["c1"]
    assert len(rels2) == 1 and rels2[0].target == "C"
    # 快照已持久化，重载后一致
    gm2 = GraphManager(snapshot_root=tmp_path)
    await gm2._load_snapshot("kb1")
    ents3, _ = gm2._graphs["kb1"]
    assert set(ents3) == {"A", "C"}


async def test_build_document_no_entity_stacking(tmp_path, monkeypatch):
    from knowlink.graph.builder import GraphManager

    gm = GraphManager(snapshot_root=tmp_path)

    async def fake_build(kb_id, knowledge_id, chunks):
        ids = [c["id"] for c in chunks]
        return {"A": Entity(entity_id="x", name="A", source_chunk_ids=ids, frequency=len(ids))}, []

    monkeypatch.setattr(gm._builder, "build", fake_build)
    await gm.build_document("kb1", "d1", [{"id": "c1", "content": "t"}])
    # 重解析：旧 chunk 软删后先对账再构建（新 chunk id 变为 c2）
    await gm.reconcile("kb1", {"c2"})
    await gm.build_document("kb1", "d1", [{"id": "c2", "content": "t"}])
    ents, _ = gm._graphs["kb1"]
    assert len(ents) == 1
    assert ents["A"].source_chunk_ids == ["c2"]
