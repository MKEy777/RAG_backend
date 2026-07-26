from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..llm.client import get_llm_client
from ..llm.json_repair import parse_json_array, parse_json_object
from ..llm.retry import wiki_llm_retry
from ..logging import bind_kb
from ..models.chunk import Chunk
from ..models.knowledge import Knowledge
from ..models.knowledge_base import KnowledgeBase
from ..settings import settings

WIKI_SUMMARY_PROMPT = (
    "You are a wiki editor. Given the following document content, create a structured wiki summary page in Markdown format.\n"
    "First line MUST be: SUMMARY: {one sentence summary, 15-40 words}\n"
    "Use ## for sections, ### for subsections.\n"
    "When available wiki pages' names appear in text, use [[slug|DisplayName]] link format.\n"
    "End with ## 关键要点 as a bullet list.\n"
    "IMPORTANT: Write ALL content in Chinese (except proper nouns, technical terms, and code).\n"
    "If content is empty, output: SUMMARY: 本文档未提取到有效文本内容。"
)

WIKI_CANDIDATE_SLUG_PROMPT = (
    "You are a knowledge extraction system. Analyze the following document and list all significant "
    "entities AND key concepts as a lightweight candidate set.\n"
    "Output JSON: {\"entities\": [...], \"concepts\": [...]}\n"
    "Entity fields: name, slug (format entity/xxx), aliases, description (one sentence), details (1-3 sentences, max 300 chars)\n"
    "Concept fields: same as entity, slug format concept/xxx\n"
    "IMPORTANT: name, description, details MUST be in Chinese (except proper nouns and technical terms). slug stays in English/pinyin.\n"
    "Empty content returns empty arrays. Do not fabricate."
)

WIKI_CITATION_PROMPT = (
    "You are a precise citation system. Scan the batch of document chunks and decide, "
    "for each candidate entity/concept below, which chunks substantively discuss it.\n"
    "Output: {\"citations\": {\"entity/xxx\": [\"chunk_id1\", \"chunk_id2\"], ...}, \"new_slugs\": []}\n"
    "Only select chunks that provide concrete facts about the candidate. Mere mentions do not count."
)


@dataclass
class MapResult:
    overview_content: str = ""
    overview_summary: str = ""
    candidates: list[dict] = field(default_factory=list)
    citations: dict[str, list[str]] = field(default_factory=dict)


async def map_document(session: Session, knowledge_id: str, kb_id: str) -> MapResult | None:
    kb = session.get(KnowledgeBase, kb_id)
    doc = session.get(Knowledge, knowledge_id)
    bind_kb(kb_id, doc_id=knowledge_id, kb_name=kb.name if kb else "", filename=doc.file_name if doc else "")
    stmt = (
        select(Chunk)
        .where(
            Chunk.knowledge_id == knowledge_id,
            Chunk.chunk_type == "text",
            Chunk.deleted_at.is_(None),
        )
        .order_by(Chunk.start_at)
    )
    chunks = session.execute(stmt).scalars().all()
    if not chunks:
        return None

    source_text = "\n".join(c.content for c in chunks)
    if not source_text.strip():
        return None
    chunk_dicts = [{"id": c.id, "content": c.content[:500]} for c in chunks]
    return await map_from_text(source_text, chunk_dicts, kb_id, knowledge_id)


async def map_from_text(
    source_text: str, chunk_dicts: list[dict], kb_id: str, knowledge_id: str
) -> MapResult | None:
    """LLM-only map phase, safe for concurrent execution (no DB access)."""
    bind_kb(kb_id, doc_id=knowledge_id)
    if not source_text.strip():
        return None
    if len(source_text) > settings.wiki_max_source_chars:
        source_text = source_text[: settings.wiki_max_source_chars]

    client = get_llm_client()
    result = MapResult()

    summary_task = _generate_summary(client, source_text)
    candidate_task = _extract_candidates(client, source_text)
    summary_result, candidates = await asyncio.gather(summary_task, candidate_task)

    result.overview_content = summary_result.get("content", "")
    result.overview_summary = summary_result.get("summary", "")
    result.candidates = candidates

    if candidates:
        result.citations = await _map_citations(client, candidates, chunk_dicts)

    return result


async def _generate_summary(client, source_text: str) -> dict:
    @wiki_llm_retry()
    async def _call():
            return await client.chat(
            [
                {"role": "system", "content": WIKI_SUMMARY_PROMPT},
                {"role": "user", "content": source_text},
            ],
            temperature=0.3,
            max_tokens=4096,
            purpose="wiki:summarize",
        )

    try:
        resp = await _call()
        lines = resp.strip().split("\n")
        summary = ""
        content = resp
        if lines and lines[0].startswith("SUMMARY:"):
            summary = lines[0][8:].strip()
            content = "\n".join(lines[1:]).strip()
        return {"summary": summary, "content": content}
    except Exception:
        return {"summary": "", "content": ""}


async def _extract_candidates(client, source_text: str) -> list[dict]:
    @wiki_llm_retry()
    async def _call():
        return await client.chat(
            [
                {"role": "system", "content": WIKI_CANDIDATE_SLUG_PROMPT},
                {"role": "user", "content": source_text},
            ],
            temperature=0.0,
            max_tokens=4096,
            purpose="wiki:extract_candidates",
        )

    try:
        resp = await _call()
        data = parse_json_object(resp)
        entities = data.get("entities", [])
        concepts = data.get("concepts", [])
        all_candidates = []
        for e in entities:
            e["_type"] = "entity"
            all_candidates.append(e)
        for c in concepts:
            c["_type"] = "concept"
            all_candidates.append(c)
        return all_candidates
    except Exception:
        return []


async def _map_citations(
    client, candidates: list[dict], chunks: list[dict]
) -> dict[str, list[str]]:
    batch_size = settings.wiki_citation_batch_chars
    sem = asyncio.Semaphore(settings.wiki_citation_concurrency)
    all_citations: dict[str, list[str]] = {}

    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_len = 0
    for chunk in chunks:
        clen = len(chunk.get("content", ""))
        if current_len + clen > batch_size and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_len = 0
        current_batch.append(chunk)
        current_len += clen
    if current_batch:
        batches.append(current_batch)

    slug_list = [c.get("slug", c.get("name", "")) for c in candidates]

    async def _process_batch(batch: list[dict]):
        async with sem:
            chunk_text = "\n".join(
                f'<chunk id="{c["id"]}">{c["content"]}</chunk>' for c in batch
            )
            user_msg = (
                f"<candidate_slugs>{slug_list}</candidate_slugs>\n"
                f"<chunks>\n{chunk_text}\n</chunks>"
            )
            try:
                @wiki_llm_retry()
                async def _call():
                    return await client.chat(
                        [
                            {"role": "system", "content": WIKI_CITATION_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.0,
                        max_tokens=2048,
                        purpose="wiki:citation",
                    )

                resp = await _call()
                data = parse_json_object(resp)
                citations = data.get("citations", {})
                for slug, chunk_ids in citations.items():
                    if slug not in all_citations:
                        all_citations[slug] = []
                    all_citations[slug].extend(chunk_ids)
            except Exception:
                pass

    await asyncio.gather(*[_process_batch(b) for b in batches])
    return all_citations
