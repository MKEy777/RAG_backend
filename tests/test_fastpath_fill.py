import numpy as np
import pytest

from knowlink.fastpath.filler import fill_answer, generate_template_from_chunk
from knowlink.models.chunk import Chunk
from knowlink.models.fastpath import FastPathEntry
from knowlink.models.knowledge import Knowledge


async def _mk_entry(db_session, template, variables=None, chunk_id="chunk1", doc_id="doc1"):
    db_session.add(Chunk(id=chunk_id, tenant_id=1, knowledge_base_id="kb1",
                         knowledge_id=doc_id, content="根据条款，等待期为30天，产品名称为守护一生",
                         chunk_type="text"))
    db_session.add(Knowledge(id=doc_id, tenant_id=1, knowledge_base_id="kb1", type="pdf",
                             file_name="doc1.pdf", parse_status="completed"))
    entry = FastPathEntry(
        kb_id="kb1", canonical_query="等待期多久", answer_template=template,
        variables=variables, target_chunk_id=chunk_id, target_document_id=doc_id,
    )
    db_session.add(entry)
    db_session.commit()
    return entry


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def chat(self, messages, model=None, temperature=0.3, max_tokens=4096, purpose=""):
        self.calls.append(purpose)
        return self.response


@pytest.mark.asyncio
async def test_static_template_no_llm_call(db_session):
    entry = await _mk_entry(db_session, "等待期为30天")
    answer = await fill_answer(db_session, entry)
    assert answer == "等待期为30天"


@pytest.mark.asyncio
async def test_variable_fill_from_target_chunk(db_session, monkeypatch):
    from knowlink.fastpath import filler as filler_mod

    entry = await _mk_entry(
        db_session,
        "根据{product_name}条款，等待期为{waiting_period}天",
        variables=[{"name": "product_name", "description": "产品名"}, {"name": "waiting_period", "description": "等待期天数"}],
    )
    fake = FakeLLM('{"product_name": "守护一生", "waiting_period": "30"}')
    monkeypatch.setattr(filler_mod, "get_llm_client", lambda: fake)
    answer = await fill_answer(db_session, entry)
    assert answer == "根据守护一生条款，等待期为30天"
    assert fake.calls == ["fastpath:fill"]


@pytest.mark.asyncio
async def test_missing_variable_falls_back(db_session, monkeypatch):
    from knowlink.fastpath import filler as filler_mod

    entry = await _mk_entry(
        db_session,
        "根据{product_name}条款，等待期为{waiting_period}天",
        variables=[{"name": "product_name"}, {"name": "waiting_period"}],
    )
    fake = FakeLLM('{"product_name": "守护一生"}')  # waiting_period 缺失
    monkeypatch.setattr(filler_mod, "get_llm_client", lambda: fake)
    assert await fill_answer(db_session, entry) is None


@pytest.mark.asyncio
async def test_deleted_target_chunk_returns_none(db_session, monkeypatch):
    from datetime import datetime, timezone

    from knowlink.fastpath import filler as filler_mod

    entry = await _mk_entry(db_session, "根据{product_name}条款", variables=[{"name": "product_name"}])
    chunk = db_session.get(Chunk, "chunk1")
    chunk.deleted_at = datetime.now(timezone.utc)
    db_session.commit()
    monkeypatch.setattr(filler_mod, "get_llm_client", lambda: FakeLLM('{"product_name": "x"}'))
    assert await fill_answer(db_session, entry) is None


@pytest.mark.asyncio
async def test_generate_template_from_chunk(db_session, monkeypatch):
    from knowlink.fastpath import filler as filler_mod

    fake = FakeLLM('{"answer_template": "等待期为{waiting_period}天", "variables": [{"name": "waiting_period", "description": "天数"}]}')
    monkeypatch.setattr(filler_mod, "get_llm_client", lambda: fake)
    template, variables = await generate_template_from_chunk("等待期多久", "等待期为30天")
    assert template == "等待期为{waiting_period}天"
    assert variables == [{"name": "waiting_period", "description": "天数"}]


@pytest.mark.asyncio
async def test_generate_template_handles_bad_json(db_session, monkeypatch):
    from knowlink.fastpath import filler as filler_mod

    fake = FakeLLM("not json at all")
    monkeypatch.setattr(filler_mod, "get_llm_client", lambda: fake)
    template, variables = await generate_template_from_chunk("q", "ctx")
    assert template == "" and variables == []
