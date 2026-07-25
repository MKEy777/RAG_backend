from __future__ import annotations

import hashlib
import socket
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

import logging as _logging

from ...logging import get_logger
from ...models.database import get_session
from ...models.knowledge import Knowledge
from ...models.knowledge_base import KnowledgeBase
from ...settings import settings
from ...tasks.parse_tasks import build_graph_task, parse_and_index_task

log = get_logger("documents")
router = APIRouter()

MAX_UPLOAD_BYTES = settings.max_upload_size_mb * 1024 * 1024

_uvicorn_log = _logging.getLogger("uvicorn")
_dbg_count = 0


def dbg(msg: str):
    global _dbg_count
    _dbg_count += 1
    _uvicorn_log.info(f"[{_dbg_count}] {msg}")


def _is_redis_available() -> bool:
    try:
        with socket.create_connection(("localhost", 6379), timeout=1):
            return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


def _run_parse_sync(knowledge_id: str, kb_id: str, tenant_id: int, file_data: bytes, filename: str):
    import asyncio
    import traceback
    from ...models.database import SessionLocal
    from ...parsing.registry import parse_file
    from ...parsing.base import ParserCapabilityError
    from ...indexing.pipeline import run_indexing_pipeline

    dbg(f"[同步解析] 开始: {filename} ({len(file_data)} bytes)")
    session = SessionLocal()
    try:
        result = parse_file(file_data, filename)
        dbg(f"[2/4] 解析完成 markdown={len(result.markdown)}")

        kb = session.get(Knowledge, knowledge_id)
        chunking_config = {}
        if kb and kb.metadata_:
            chunking_config = kb.metadata_.get("chunking_config", {})

        dbg("[3/4] 切块和嵌入")
        asyncio.run(
            run_indexing_pipeline(
                session=session,
                knowledge_id=knowledge_id,
                kb_id=kb_id,
                tenant_id=tenant_id,
                markdown=result.markdown,
                doc_title=filename,
                chunking_config=chunking_config,
            )
        )
        dbg("[4/4] 完成")
    except ParserCapabilityError as e:
        session.execute(
            Knowledge.__table__.update()
            .where(Knowledge.id == knowledge_id)
            .values(parse_status="failed", error_message=str(e))
        )
        session.commit()
        dbg(f"[失败] 格式不支持: {e}")
    except Exception as e:
        tb = traceback.format_exc()
        session.execute(
            Knowledge.__table__.update()
            .where(Knowledge.id == knowledge_id)
            .values(parse_status="failed", error_message=tb[:2000])
        )
        session.commit()
        dbg(f"[异常] {e}")
        dbg(tb)
    finally:
        session.close()


def _run_graph_sync(kb_id: str, knowledge_id: str):
    import asyncio
    import traceback
    from sqlalchemy import select
    from ...models.database import SessionLocal
    from ...models.chunk import Chunk
    from ...graph.builder import get_graph_manager

    dbg("[图谱] 开始构建")
    session = SessionLocal()
    try:
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
        dbg(f"[图谱] 加载切块: {len(chunks)} 个")
        chunk_dicts = [{"id": c.id, "content": c.content} for c in chunks]
        gm = get_graph_manager()
        asyncio.run(gm.build_document(kb_id, knowledge_id, chunk_dicts))
        dbg("[图谱] 构建完成")
    except Exception as e:
        dbg(f"[图谱] 失败: {e}")
        traceback.print_exc()
    finally:
        session.close()


class DocResponse(BaseModel):
    id: str
    knowledge_base_id: str
    title: str | None
    file_name: str | None
    parse_status: str

    class Config:
        from_attributes = True


@router.post("/upload", response_model=DocResponse)
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    dbg(f"收到上传请求 kb_id={kb_id} file={file.filename}")
    kb = session.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")

    filename = file.filename or "untitled"
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(415, f"File type '{ext}' not allowed. Allowed: {settings.allowed_extensions}")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Max {settings.max_upload_size_mb}MB")

    file_hash = hashlib.sha256(data).hexdigest()

    existing = (
        session.query(Knowledge)
        .filter(
            Knowledge.knowledge_base_id == kb_id,
            Knowledge.file_hash == file_hash,
            Knowledge.deleted_at.is_(None),
        )
        .first()
    )
    if existing:
        log.info("上传去重_文件已存在", kb_id=kb_id, file_hash=file_hash[:12], doc_id=existing.id)
        return existing

    doc = Knowledge(
        id=uuid.uuid4().hex,
        tenant_id=kb.tenant_id,
        knowledge_base_id=kb_id,
        type="file",
        title=filename,
        file_name=filename,
        file_type=file.content_type or "",
        file_size=len(data),
        file_hash=file_hash,
        source="upload",
        parse_status="processing",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    log.info("上传成功", kb_id=kb_id, doc_id=doc.id, filename=filename, size=len(data))

    redis_ok = _is_redis_available()

    if redis_ok:
        parse_and_index_task.delay(doc.id, kb_id, kb.tenant_id, data, filename)
        build_graph_task.delay(kb_id, doc.id)
        log.info("使用 Celery 异步处理", doc_id=doc.id)
    else:
        import threading

        def _sync_process():
            _run_parse_sync(doc.id, kb_id, kb.tenant_id, data, filename)
            _run_graph_sync(kb_id, doc.id)

        threading.Thread(target=_sync_process, daemon=True).start()
        log.info("使用同步处理（Redis 不可用）", doc_id=doc.id)

    return doc


@router.get("")
def list_documents(
    kb_id: str,
    page: int = 1,
    size: int = 20,
    session: Session = Depends(get_session),
):
    query = (
        session.query(Knowledge)
        .filter(Knowledge.knowledge_base_id == kb_id, Knowledge.deleted_at.is_(None))
    )
    total = query.count()
    items = query.offset((page - 1) * size).limit(size).all()
    return {
        "items": [DocResponse.model_validate(d) for d in items],
        "total": total,
        "page": page,
        "size": size,
    }


@router.get("/{doc_id}", response_model=DocResponse)
def get_document(doc_id: str, session: Session = Depends(get_session)):
    doc = session.get(Knowledge, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


@router.get("/{doc_id}/status")
def get_document_status(doc_id: str, session: Session = Depends(get_session)):
    import json

    doc = session.get(Knowledge, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    progress = {"stage": doc.parse_status, "percent": 100 if doc.parse_status == "completed" else 0}
    try:
        import redis as redis_lib

        r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=2)
        raw = r.get(f"task_progress:{doc_id}")
        if raw:
            progress = json.loads(raw)
    except Exception:
        pass

    return {
        "doc_id": doc_id,
        "parse_status": doc.parse_status,
        "progress": progress,
        "error_message": doc.error_message,
    }


@router.delete("/{doc_id}")
def delete_document(doc_id: str, session: Session = Depends(get_session)):
    from datetime import datetime, timezone

    doc = session.get(Knowledge, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    doc.deleted_at = datetime.now(timezone.utc)
    session.commit()
    return {"status": "deleted"}
