from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict
from pathlib import Path

from ..logging import bind_kb, get_logger
from ..llm.client import get_llm_client
from ..llm.json_repair import parse_json_array
from ..llm.retry import graph_llm_retry
from ..prompts.loader import get_prompt
from ..settings import settings
from .weights import Entity, Relation, compute_degrees, compute_pmi_weights

log = get_logger("graph")

GRAPH_PROMPT_LANGUAGE = "中文"  # detail.md 中文场景；实体 title 保持原文语言

# YAML 缺失时的降级提示词（完整版在 config/prompts/graph.yaml，移植自 WeKnora graph_extraction.yaml）
EXTRACT_ENTITIES_PROMPT = (
    "Extract all entities from the user-provided text that match the following entity types: "
    "Person, Organization, Location, Product, Event, Date, Work, Concept, Resource, Category, Operation.\n"
    "Output a JSON array, each item: {\"title\": \"name\", \"type\": \"type\", \"description\": \"one sentence\"}.\n"
    "Extract entity titles exactly as they appear in the source text; write descriptions in Chinese.\n"
    "Only output JSON, no extra text."
)

EXTRACT_RELATIONS_PROMPT = (
    "From the user-provided entity array, extract explicit relationships between entities.\n"
    "Output a JSON array, each item: {\"source\": \"entity title\", \"target\": \"entity title\", "
    "\"description\": \"relationship description\", \"strength\": strength(5-10)}.\n"
    "Write descriptions in Chinese. Only output JSON, no extra text."
)


def _entities_prompt() -> str:
    return get_prompt("graph", "default_extract_entities", {"language": GRAPH_PROMPT_LANGUAGE}) or EXTRACT_ENTITIES_PROMPT


def _relations_prompt() -> str:
    return get_prompt("graph", "default_extract_relationships", {"language": GRAPH_PROMPT_LANGUAGE}) or EXTRACT_RELATIONS_PROMPT


class GraphBuilder:
    async def build(
        self, kb_id: str, knowledge_id: str, chunks: list[dict]
    ) -> tuple[dict[str, Entity], list[Relation]]:
        entity_map: dict[str, Entity] = {}
        relations: list[Relation] = []

        log.info("图谱构建开始", kb_id=kb_id, knowledge_id=knowledge_id, chunk_count=len(chunks))
        await self._extract_entities(chunks, entity_map)
        await self._extract_relations(chunks, entity_map, relations)

        compute_pmi_weights(entity_map, relations)
        compute_degrees(entity_map, relations)
        log.info("图谱构建完成", kb_id=kb_id, entities=len(entity_map), relations=len(relations))
        return entity_map, relations

    async def _extract_entities(
        self, chunks: list[dict], entity_map: dict[str, Entity]
    ) -> None:
        sem = asyncio.Semaphore(settings.graph_entity_concurrency)
        client = get_llm_client()
        fail_count = 0

        async def _process(chunk: dict):
            nonlocal fail_count
            async with sem:
                text = chunk.get("content", "")
                if not text.strip():
                    return
                try:
                    @graph_llm_retry()
                    async def _call():
                        return await client.chat(
                            [
                                {"role": "system", "content": _entities_prompt()},
                                {"role": "user", "content": text},
                            ],
                            temperature=0.3,
                            max_tokens=2048,
                            purpose="graph:extract_entities",
                        )

                    resp = await _call()
                    items = parse_json_array(resp)
                    chunk_id = chunk.get("id", "")
                    for item in items:
                        title = str(item.get("title", "")).strip()
                        desc = str(item.get("description", "")).strip()
                        if not title or not desc:
                            continue
                        if title in entity_map:
                            ent = entity_map[title]
                            if chunk_id and chunk_id not in ent.source_chunk_ids:
                                ent.source_chunk_ids.append(chunk_id)
                            ent.frequency += 1
                        else:
                            entity_map[title] = Entity(
                                entity_id=uuid.uuid4().hex,
                                name=title,
                                description=desc,
                                type=str(item.get("type", "")),
                                source_chunk_ids=[chunk_id] if chunk_id else [],
                                frequency=1,
                            )
                except Exception as e:
                    fail_count += 1
                    log.warning("实体提取失败", chunk_id=chunk.get("id"), error=str(e)[:200])

        await asyncio.gather(*[_process(c) for c in chunks])
        if fail_count:
            log.warning("实体提取部分失败", total=len(chunks), failed=fail_count)

    async def _extract_relations(
        self,
        chunks: list[dict],
        entity_map: dict[str, Entity],
        relations: list[Relation],
    ) -> None:
        batch_size = settings.graph_relation_batch_size
        sem = asyncio.Semaphore(settings.graph_relation_concurrency)
        client = get_llm_client()
        relation_map: dict[str, Relation] = {}

        batches = [
            chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)
        ]
        log.info("关系提取开始", batch_count=len(batches), batch_size=batch_size)
        fail_count = 0

        async def _process_batch(batch: list[dict]):
            nonlocal fail_count
            async with sem:
                batch_entities: set[str] = set()
                for chunk in batch:
                    chunk_id = chunk.get("id", "")
                    for name, ent in entity_map.items():
                        if chunk_id in ent.source_chunk_ids:
                            batch_entities.add(name)

                if len(batch_entities) < 2:
                    return

                combined_text = "\n".join(c.get("content", "") for c in batch)
                ent_list = [
                    {"title": n, "description": entity_map[n].description}
                    for n in batch_entities
                ]
                user_msg = (
                    f"Entities:\n{json.dumps(ent_list, ensure_ascii=False)}\n\n"
                    f"Text:\n{combined_text}"
                )

                try:
                    @graph_llm_retry()
                    async def _call():
                        return await client.chat(
                            [
                                {"role": "system", "content": _relations_prompt()},
                                {"role": "user", "content": user_msg},
                            ],
                            temperature=0.0,
                            max_tokens=2048,
                            purpose="graph:extract_relations",
                        )

                    resp = await _call()
                    items = parse_json_array(resp)

                    for item in items:
                        src = str(item.get("source", "")).strip()
                        tgt = str(item.get("target", "")).strip()
                        if not src or not tgt or src not in batch_entities or tgt not in batch_entities:
                            continue
                        key = f"{src}#{tgt}"
                        strength = float(item.get("strength", 5))
                        desc = str(item.get("description", ""))

                        # 关系归属于两实体的 source_chunk_ids 并集（WeKnora findRelationChunkIDs），而非整批 chunk
                        rel_chunk_ids = sorted(
                            set(entity_map[src].source_chunk_ids) | set(entity_map[tgt].source_chunk_ids)
                        )
                        if not rel_chunk_ids:
                            continue

                        if key in relation_map:
                            rel = relation_map[key]
                            added = 0
                            for cid in rel_chunk_ids:
                                if cid not in rel.source_chunk_ids:
                                    rel.source_chunk_ids.append(cid)
                                    added += 1
                            # strength 加权平均（WeKnora 同款），不直接覆盖
                            rel.strength = (rel.strength * len(rel.source_chunk_ids) + strength) / (
                                len(rel.source_chunk_ids) + 1
                            )
                            rel.rel_freq = len(rel.source_chunk_ids)
                            if added:
                                rel.description = desc
                        else:
                            relation_map[key] = Relation(
                                source=src,
                                target=tgt,
                                description=desc,
                                strength=strength,
                                rel_freq=len(rel_chunk_ids),
                                source_chunk_ids=rel_chunk_ids,
                            )
                except Exception as e:
                    fail_count += 1
                    log.warning("关系提取失败", error=str(e)[:200])

        await asyncio.gather(*[_process_batch(b) for b in batches])
        relations.extend(relation_map.values())
        log.info("关系提取完成", count=len(relation_map), failed_batches=fail_count)


class GraphManager:
    def __init__(self, snapshot_root: Path | None = None):
        self._root = snapshot_root or settings.object_storage_root / "graphs"
        self._root.mkdir(parents=True, exist_ok=True)
        self._graphs: dict[str, tuple[dict[str, Entity], list[Relation]]] = {}
        self._builder = GraphBuilder()

    async def build_document(
        self, kb_id: str, knowledge_id: str, chunks: list[dict]
    ) -> None:
        bind_kb(kb_id, doc_id=knowledge_id)
        log.info("图管理器构建文档", kb_id=kb_id, knowledge_id=knowledge_id, chunk_count=len(chunks))
        entity_map, relations = await self._builder.build(kb_id, knowledge_id, chunks)

        if kb_id not in self._graphs:
            await self._load_snapshot(kb_id)
        if kb_id not in self._graphs:
            self._graphs[kb_id] = ({}, [])
        existing_ents, existing_rels = self._graphs[kb_id]
        # 同名实体合并 chunk 来源与频次，而非整体覆盖（跨文档共享实体）
        for name, ent in entity_map.items():
            if name in existing_ents:
                exist = existing_ents[name]
                for cid in ent.source_chunk_ids:
                    if cid not in exist.source_chunk_ids:
                        exist.source_chunk_ids.append(cid)
                exist.frequency += ent.frequency
            else:
                existing_ents[name] = ent
        existing_keys = {f"{r.source}#{r.target}" for r in existing_rels}
        for rel in relations:
            if f"{rel.source}#{rel.target}" not in existing_keys:
                existing_rels.append(rel)

        compute_pmi_weights(existing_ents, existing_rels)
        compute_degrees(existing_ents, existing_rels)
        await self._save_snapshot(kb_id)
        log.info("图管理器文档构建完成", kb_id=kb_id, total_entities=len(existing_ents),
                 total_relations=len(existing_rels))

    async def reconcile(self, kb_id: str, live_chunk_ids: set[str]) -> int:
        """以 DB 活块集为准清理死 chunk 引用；旧版按 knowledge_id 前缀匹配从不命中，导致实体只增不减。"""
        if kb_id not in self._graphs:
            await self._load_snapshot(kb_id)
        if kb_id not in self._graphs:
            return 0
        ents, rels = self._graphs[kb_id]
        removed = 0
        for name in list(ents.keys()):
            ent = ents[name]
            kept = [cid for cid in ent.source_chunk_ids if cid in live_chunk_ids]
            if len(kept) != len(ent.source_chunk_ids):
                ent.source_chunk_ids = kept
                ent.frequency = max(len(kept), 1)
            if not kept:
                del ents[name]
                removed += 1

        new_rels = []
        for rel in rels:
            if rel.source not in ents or rel.target not in ents:
                continue
            rel.source_chunk_ids = [cid for cid in rel.source_chunk_ids if cid in live_chunk_ids]
            if not rel.source_chunk_ids:
                continue
            rel.rel_freq = len(rel.source_chunk_ids)
            new_rels.append(rel)

        if removed or len(new_rels) != len(rels):
            self._graphs[kb_id] = (ents, new_rels)
            compute_pmi_weights(ents, new_rels)
            compute_degrees(ents, new_rels)
            await self._save_snapshot(kb_id)
            log.info("图谱对账", kb_id=kb_id, removed_entities=removed,
                     removed_relations=len(rels) - len(new_rels))
        return removed

    async def reset(self, kb_id: str) -> None:
        self._graphs[kb_id] = ({}, [])
        await self._save_snapshot(kb_id)
        log.info("图谱重置", kb_id=kb_id)

    async def search_by_entities(
        self, kb_id: str, entity_names: list[str], top_k: int = 10
    ) -> list[dict]:
        if kb_id not in self._graphs:
            await self._load_snapshot(kb_id)
        if kb_id not in self._graphs:
            return []

        ents, rels = self._graphs[kb_id]
        chunk_scores: dict[str, float] = {}

        matched_entities = set()
        for name in entity_names:
            if name in ents:
                matched_entities.add(name)
            for ent_name in ents:
                if name.lower() in ent_name.lower():
                    matched_entities.add(ent_name)

        for rel in rels:
            if rel.source in matched_entities or rel.target in matched_entities:
                for cid in rel.source_chunk_ids:
                    chunk_scores[cid] = chunk_scores.get(cid, 0.0) + rel.weight

        for name in matched_entities:
            ent = ents[name]
            for cid in ent.source_chunk_ids:
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + ent.degree * 0.5

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"chunk_id": cid, "score": score, "knowledge_id": kb_id}
            for cid, score in sorted_chunks[:top_k]
        ]

    async def _save_snapshot(self, kb_id: str) -> None:
        if kb_id not in self._graphs:
            return
        ents, rels = self._graphs[kb_id]
        data = {
            "entities": {n: asdict(e) for n, e in ents.items()},
            "relations": [asdict(r) for r in rels],
        }
        path = self._root / f"{kb_id}.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    async def _load_snapshot(self, kb_id: str) -> None:
        path = self._root / f"{kb_id}.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ents = {}
            for name, ed in data.get("entities", {}).items():
                ents[name] = Entity(**ed)
            rels = [Relation(**rd) for rd in data.get("relations", [])]
            self._graphs[kb_id] = (ents, rels)
        except Exception:
            pass


_manager: GraphManager | None = None


def get_graph_manager() -> GraphManager:
    global _manager
    if _manager is None:
        _manager = GraphManager()
    return _manager
