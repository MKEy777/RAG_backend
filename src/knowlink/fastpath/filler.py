from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy.orm import Session

from ..llm.client import get_llm_client
from ..llm.json_repair import parse_json_object
from ..logging import get_logger
from ..models.chunk import Chunk
from ..models.fastpath import FastPathEntry
from ..models.knowledge import Knowledge
from ..prompts.loader import get_prompt
from ..settings import settings

log = get_logger("fastpath")

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
_SOURCE_CHARS_LIMIT = 4000


def _load_target_content(session: Session, entry: FastPathEntry) -> str | None:
    """取目标 chunk 正文（含父块展开），不可用返回 None。"""
    if not entry.target_chunk_id:
        return None
    chunk = session.get(Chunk, entry.target_chunk_id)
    if chunk is None or chunk.deleted_at is not None or not chunk.enabled:
        return None
    if entry.target_document_id:
        knowledge = session.get(Knowledge, entry.target_document_id)
        if knowledge is not None and knowledge.deleted_at is not None:
            return None
    content = chunk.content
    if chunk.parent_chunk_id:
        parent = session.get(Chunk, chunk.parent_chunk_id)
        if parent and parent.deleted_at is None:
            content = parent.content
    return content[: _SOURCE_CHARS_LIMIT]


async def _extract_variables(
    content: str, variables: list[dict]
) -> dict[str, Any] | None:
    schema: dict[str, str] = {}
    for v in variables:
        name = v.get("name")
        if name:
            schema[name] = str(v.get("description", ""))
    system = get_prompt("fastpath", "variable_extraction")
    user = (
        f"<variables schema>\n{json.dumps(schema, ensure_ascii=False)}\n</variables>\n\n"
        f"<source>\n{content}\n</source>"
    )
    client = get_llm_client()
    try:
        resp = await client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=settings.fastpath_fill_max_tokens,
            purpose="fastpath:fill",
        )
        values = parse_json_object(resp)
    except Exception as e:
        log.warning("快路径变量取数失败", error=str(e))
        return None
    if not isinstance(values, dict):
        return None
    return {str(k): v for k, v in values.items()}


async def fill_answer(session: Session, entry: FastPathEntry) -> str | None:
    """模板 + 动态取数：无变量模板直接返回；有变量则从目标 chunk 抽取填充。

    任一变量缺失或取数失败返回 None，调用方回落到 L1/L2。
    """
    template = entry.answer_template or ""
    placeholders = set(_PLACEHOLDER_RE.findall(template))
    if not placeholders:
        return template

    content = _load_target_content(session, entry)
    if content is None:
        return None

    variables = entry.variables or [{"name": p} for p in placeholders]
    values = await _extract_variables(content, variables)
    if values is None or not all(p in values for p in placeholders):
        log.info("快路径变量缺失_回落完整链路", entry_id=entry.id, placeholders=placeholders)
        return None
    try:
        return template.format(**{k: str(values[k]) for k in placeholders})
    except (KeyError, ValueError) as e:
        log.warning("快路径模板填充失败", error=str(e))
        return None


async def generate_template_from_chunk(question: str, content: str) -> tuple[str, list[dict]]:
    """从问答对 + 目标正文生成答案模板与变量 schema（FAQ 升级 / 挖掘共用）。"""
    system = get_prompt("fastpath", "template_generation")
    user = f"Question: {question}\n\n<source>\n{content[: _SOURCE_CHARS_LIMIT]}\n</source>"
    client = get_llm_client()
    try:
        resp = await client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=settings.fastpath_fill_max_tokens,
            purpose="fastpath:template",
        )
        obj = parse_json_object(resp)
    except Exception as e:
        log.warning("快路径模板生成失败", error=str(e))
        return "", []
    if not isinstance(obj, dict):
        return "", []
    template = str(obj.get("answer_template", "")).strip()
    variables = obj.get("variables")
    if not isinstance(variables, list):
        variables = []
    return template, [v for v in variables if isinstance(v, dict) and v.get("name")]
