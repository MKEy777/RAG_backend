from __future__ import annotations

import hashlib
import socket
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from datetime import datetime

import logging as _logging

from ...logging import bind_kb, get_logger
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


_redis_available = _is_redis_available()


def _run_parse_sync(knowledge_id: str, kb_id: str, tenant_id: int, file_data: bytes, filename: str, base_dir: str | None = None):
    import asyncio
    import traceback
    from ...models.database import SessionLocal
    from ...parsing.registry import parse_file
    from ...parsing.base import ParserCapabilityError
    from ...indexing.pipeline import run_indexing_pipeline
    from ...indexing.image_pipeline import enrich_markdown_with_images

    from ...tasks.parse_tasks import _update_progress

    dbg(f"[同步解析] 开始: {filename} ({len(file_data)} bytes)")
    try:
        _update_progress(knowledge_id, "parsing", 10)
        result = parse_file(file_data, filename, base_dir=base_dir)
        dbg(f"[解析] 完成 markdown={len(result.markdown)} images={len(result.images)}")

        session = SessionLocal()
        try:
            doc = session.get(Knowledge, knowledge_id)
            kb_row = session.get(KnowledgeBase, kb_id)
            doc_meta = (doc.metadata_ or {}) if doc else {}
            # 文档级 > KB 级（agents.md：KB 列 chunking_config/image_processing_config 的读取点）
            chunking_config = doc_meta.get("chunking_config") or (kb_row.chunking_config if kb_row else {}) or {}
            parse_config = doc_meta.get("parse_config") or (kb_row.image_processing_config if kb_row else None) or {}
        finally:
            session.close()

        # 文本立即可索引，不等 VLM
        if result.markdown.strip():
            session = SessionLocal()
            try:
                dbg("[索引] 文本切块+嵌入")
                _update_progress(knowledge_id, "chunking", 30)
                asyncio.run(
                    run_indexing_pipeline(
                        session=session,
                        knowledge_id=knowledge_id,
                        kb_id=kb_id,
                        tenant_id=tenant_id,
                        markdown=result.markdown,
                        doc_title=filename,
                        chunking_config=chunking_config,
                        mark_completed=not result.images,
                    )
                )
                dbg("[索引] 文本完成，文档可检索")
                _update_progress(knowledge_id, "embedding", 60 if result.images else 100)
            finally:
                session.close()

        # VLM 图片处理（耗时操作，不阻塞文本检索）
        if result.images:
            _update_progress(knowledge_id, "vlm", 70)
            full_markdown = asyncio.run(enrich_markdown_with_images(result.markdown, result.images, parse_config=parse_config))
            if len(full_markdown) > len(result.markdown):
                session = SessionLocal()
                try:
                    dbg(f"[索引] 图片内容替换完成 ({len(full_markdown)} chars)")
                    asyncio.run(
                        run_indexing_pipeline(
                            session=session,
                            knowledge_id=knowledge_id,
                            kb_id=kb_id,
                            tenant_id=tenant_id,
                            markdown=full_markdown,
                            doc_title=filename,
                            chunking_config=chunking_config,
                            mark_completed=True,
                        )
                    )
                    dbg("[索引] 图片索引完成")
                finally:
                    session.close()
            else:
                # VLM 未提取到有效内容，直接标记完成
                session = SessionLocal()
                try:
                    from sqlalchemy import update as sa_update
                    from ...models.knowledge import Knowledge as K
                    session.execute(
                        K.__table__.update()
                        .where(K.id == knowledge_id)
                        .values(parse_status="completed")
                    )
                    session.commit()
                finally:
                    session.close()

        dbg("[解析管线] 全部完成")
        _update_progress(knowledge_id, "completed", 100)
    except ParserCapabilityError as e:
        _update_progress(knowledge_id, "failed", 0)
        session = SessionLocal()
        try:
            session.execute(
                Knowledge.__table__.update()
                .where(Knowledge.id == knowledge_id)
                .values(parse_status="failed", error_message=str(e))
            )
            session.commit()
        finally:
            session.close()
        dbg(f"[失败] 格式不支持: {e}")
    except Exception as e:
        _update_progress(knowledge_id, "failed", 0)
        tb = traceback.format_exc()
        session = SessionLocal()
        try:
            session.execute(
                Knowledge.__table__.update()
                .where(Knowledge.id == knowledge_id)
                .values(parse_status="failed", error_message=tb[:2000])
            )
            session.commit()
        finally:
            session.close()
        dbg(f"[异常] {e}")
        dbg(tb)


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
        session.execute(
            Knowledge.__table__.update()
            .where(Knowledge.id == knowledge_id)
            .values(graph_status="processing")
        )
        session.commit()

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
        # 重解析场景：先清掉旧版已软删 chunk 的实体引用，再合并新结果
        from ...indexing.reconcile import live_text_chunk_ids

        asyncio.run(gm.reconcile(kb_id, live_text_chunk_ids(session, kb_id)))
        asyncio.run(gm.build_document(kb_id, knowledge_id, chunk_dicts))

        session.execute(
            Knowledge.__table__.update()
            .where(Knowledge.id == knowledge_id)
            .values(graph_status="completed")
        )
        session.commit()
        dbg("[图谱] 构建完成")
    except Exception as e:
        session.execute(
            Knowledge.__table__.update()
            .where(Knowledge.id == knowledge_id)
            .values(graph_status="failed")
        )
        session.commit()
        dbg(f"[图谱] 失败: {e}")
        traceback.print_exc()
    finally:
        session.close()


def _run_wiki_sync(kb_id: str, knowledge_id: str):
    import traceback
    from ...models.database import SessionLocal
    from ...wiki.build import run_wiki_build
    from ...wiki.queue import enqueue_ingest

    dbg("[Wiki] 开始构建")
    session = SessionLocal()
    try:
        session.execute(
            Knowledge.__table__.update()
            .where(Knowledge.id == knowledge_id)
            .values(wiki_status="processing")
        )
        session.commit()

        # 显式入队本文档：重解析时同样触发 wiki 更新（replace 语义）
        enqueue_ingest(session, kb_id, knowledge_id)
        run_wiki_build(session, kb_id)

        session.execute(
            Knowledge.__table__.update()
            .where(Knowledge.id == knowledge_id)
            .values(wiki_status="completed")
        )
        session.commit()
        dbg("[Wiki] 构建完成")
    except Exception as e:
        session.execute(
            Knowledge.__table__.update()
            .where(Knowledge.id == knowledge_id)
            .values(wiki_status="failed")
        )
        session.commit()
        dbg(f"[Wiki] 失败: {e}")
        traceback.print_exc()
    finally:
        session.close()


class DocResponse(BaseModel):
    id: str
    knowledge_base_id: str
    title: str | None
    file_name: str | None
    parse_status: str
    needs_images: bool = False
    file_size: int | None = None
    file_type: str | None = None
    file_hash: str | None = None
    graph_status: str | None = None
    wiki_status: str | None = None
    error_message: str | None = None
    tags: list | None = None
    uploaded_by: str | None = None
    authority: int = 0
    wiki_enabled: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


class ChunkResponse(BaseModel):
    id: str
    content: str
    chunk_type: str
    parent_chunk_id: str | None
    start_at: int | None
    end_at: int | None
    enabled: bool = True
    created_at: datetime | None

    class Config:
        from_attributes = True


@router.post("/upload", response_model=DocResponse)
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    display_name: str | None = Form(None),
    tags: str | None = Form(None),
    uploaded_by: str | None = Form(None),
    parse_config: str | None = Form(None),
    chunking_config: str | None = Form(None),
    base_dir: str | None = None,
    session: Session = Depends(get_session),
):
    dbg(f"收到上传请求 kb_id={kb_id} file={file.filename}")
    kb = session.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")

    import json as _json

    def _parse_json_form(raw: str | None, name: str) -> dict | None:
        if not raw:
            return None
        try:
            parsed = _json.loads(raw)
        except ValueError:
            raise HTTPException(422, f"{name} 必须是合法 JSON 对象")
        if not isinstance(parsed, dict):
            raise HTTPException(422, f"{name} 必须是合法 JSON 对象")
        return parsed

    parse_cfg = _parse_json_form(parse_config, "parse_config")
    chunking_cfg = _parse_json_form(chunking_config, "chunking_config")

    tag_list: list | None = None
    if tags:
        try:
            parsed = _json.loads(tags)
            if isinstance(parsed, list):
                tag_list = [str(t) for t in parsed if str(t).strip()]
        except ValueError:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

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

    # 检测 md 文件中是否有未解析的相对路径图片引用
    needs_images = False
    if ext in (".md", ".markdown") and not base_dir:
        import re as _re
        text_preview = data.decode("utf-8", errors="replace")
        has_md_img = _re.search(r"!\[[^\]]*\]\((?!http|data:|images/)[^)]+\)", text_preview)
        has_html_img = _re.search(r"""<img\s+[^>]*src=["'](?!http|data:|images/)[^"']+["']""", text_preview, _re.IGNORECASE)
        needs_images = bool(has_md_img or has_html_img)

    doc_id = uuid.uuid4().hex
    original_key = f"originals/{kb_id}/{doc_id}/{filename}"
    meta: dict = {"original_key": original_key, "original_filename": filename}
    if needs_images:
        meta["needs_images"] = True
    if parse_cfg:
        meta["parse_config"] = parse_cfg
    if chunking_cfg:
        meta["chunking_config"] = chunking_cfg

    doc = Knowledge(
        id=doc_id,
        tenant_id=kb.tenant_id,
        knowledge_base_id=kb_id,
        type="file",
        title=display_name or filename,
        file_name=display_name or filename,
        file_type=file.content_type or "",
        file_size=len(data),
        file_hash=file_hash,
        source="upload",
        parse_status="processing",
        tags=tag_list,
        uploaded_by=uploaded_by or "admin",
        metadata_=meta,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    try:
        from ...storage.factory import get_storage

        await get_storage().objects.put(original_key, data)
    except Exception as e:
        log.warning("原文件存档失败", doc_id=doc.id, error=str(e)[:200])

    log.info("上传成功", kb_id=kb_id, doc_id=doc.id, filename=filename, size=len(data))
    if needs_images:
        log.info("检测到本地图片引用", doc_id=doc.id, filename=filename)

    _dispatch_processing(kb, doc.id, data, filename, base_dir=base_dir)

    return DocResponse(
        id=doc.id,
        knowledge_base_id=doc.knowledge_base_id,
        title=doc.title,
        file_name=doc.file_name,
        parse_status=doc.parse_status,
        needs_images=needs_images,
        tags=doc.tags,
        uploaded_by=doc.uploaded_by,
    )


def _dispatch_processing(kb: KnowledgeBase, doc_id: str, data: bytes, filename: str, base_dir: str | None = None):
    kb_id = kb.id
    graph_enabled = (kb.graph_config or {}).get("enabled", True)
    wiki_enabled = (kb.wiki_config or {}).get("enabled", True)

    if _redis_available:
        parse_and_index_task.delay(doc_id, kb_id, kb.tenant_id, data, filename)
        if graph_enabled:
            build_graph_task.delay(kb_id, doc_id)
        log.info("使用 Celery 异步处理", doc_id=doc_id)
    else:
        import threading

        def _sync_process():
            bind_kb(kb_id, doc_id=doc_id, kb_name=kb.name, filename=filename)
            _run_parse_sync(doc_id, kb_id, kb.tenant_id, data, filename, base_dir=base_dir)

            # 图谱和 Wiki 互不依赖，并行执行
            post_threads = []
            if graph_enabled:
                t = threading.Thread(target=_run_graph_sync, args=(kb_id, doc_id), daemon=True)
                t.start()
                post_threads.append(t)
            if wiki_enabled:
                t = threading.Thread(target=_run_wiki_sync, args=(kb_id, doc_id), daemon=True)
                t.start()
                post_threads.append(t)
            for t in post_threads:
                t.join()

        threading.Thread(target=_sync_process, daemon=True).start()
        log.info("使用同步处理（Redis 不可用）", doc_id=doc_id)


async def _save_and_dispatch(doc_id: str, kb_id: str, tenant_id: int, data: bytes, filename: str, original_key: str) -> None:
    try:
        from ...storage.factory import get_storage

        await get_storage().objects.put(original_key, data)
    except Exception as e:
        log.warning("原文件存档失败", doc_id=doc_id, error=str(e)[:200])

    kb = None
    from ...models.database import SessionLocal
    s = SessionLocal()
    try:
        kb = s.get(KnowledgeBase, kb_id)
    finally:
        s.close()
    if kb:
        _dispatch_processing(kb, doc_id, data, filename)


class LocalImportBody(BaseModel):
    kb_id: str
    file_path: str


@router.post("/import-local", response_model=DocResponse)
def import_local_document(
    body: LocalImportBody,
    session: Session = Depends(get_session),
):
    """Import a file from local filesystem path (local deployment only)."""
    file_path = Path(body.file_path)
    if not file_path.is_file():
        raise HTTPException(404, f"File not found: {body.file_path}")

    kb = session.get(KnowledgeBase, body.kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")

    filename = file_path.name
    ext = file_path.suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(415, f"File type '{ext}' not allowed. Allowed: {settings.allowed_extensions}")

    data = file_path.read_bytes()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Max {settings.max_upload_size_mb}MB")

    file_hash = hashlib.sha256(data).hexdigest()
    existing = (
        session.query(Knowledge)
        .filter(
            Knowledge.knowledge_base_id == body.kb_id,
            Knowledge.file_hash == file_hash,
            Knowledge.deleted_at.is_(None),
        )
        .first()
    )
    if existing:
        log.info("导入去重_文件已存在", kb_id=body.kb_id, file_hash=file_hash[:12], doc_id=existing.id)
        return existing

    doc = Knowledge(
        id=uuid.uuid4().hex,
        tenant_id=kb.tenant_id,
        knowledge_base_id=body.kb_id,
        type="file",
        title=filename,
        file_name=filename,
        file_type="",
        file_size=len(data),
        file_hash=file_hash,
        source="local_import",
        parse_status="processing",
        uploaded_by="admin",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    # base_dir 自动推导为文件所在目录
    base_dir = str(file_path.parent)
    log.info("本地导入成功", kb_id=body.kb_id, doc_id=doc.id, filename=filename, base_dir=base_dir)

    graph_enabled = (kb.graph_config or {}).get("enabled", True)
    wiki_enabled = (kb.wiki_config or {}).get("enabled", True)

    import threading

    def _sync_process():
        bind_kb(body.kb_id, doc_id=doc.id, kb_name=kb.name, filename=filename)
        _run_parse_sync(doc.id, body.kb_id, kb.tenant_id, data, filename, base_dir=base_dir)

        post_threads = []
        if graph_enabled:
            t = threading.Thread(target=_run_graph_sync, args=(body.kb_id, doc.id), daemon=True)
            t.start()
            post_threads.append(t)
        if wiki_enabled:
            t = threading.Thread(target=_run_wiki_sync, args=(body.kb_id, doc.id), daemon=True)
            t.start()
            post_threads.append(t)
        for t in post_threads:
            t.join()

    threading.Thread(target=_sync_process, daemon=True).start()
    return doc


class ChunkPreviewBody(BaseModel):
    text: str
    strategy: str = "auto"
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    enable_parent_child: bool = True
    separators: list[str] | None = None


@router.post("/chunk-preview")
def chunk_preview(body: ChunkPreviewBody):
    from ...chunking import chunk_document
    from ...chunking.parent_child import ParentChildPair

    result = chunk_document(
        body.text, strategy=body.strategy, chunk_size=body.chunk_size,
        overlap=body.chunk_overlap, enable_parent_child=body.enable_parent_child,
        separators=body.separators,
    )
    if result and isinstance(result[0], ParentChildPair):
        return {"mode": "parent_child", "parents": [
            {
                "text": p.parent_text[:2000],
                "title_path": p.parent_title_path,
                "children": [{"text": c.text, "start": c.start, "end": c.end} for c in p.children],
            }
            for p in result
        ]}
    return {"mode": "flat", "chunks": [
        {"text": c.text, "start": c.start, "end": c.end, "title_path": c.title_path} for c in result
    ]}


class ImportUrlBody(BaseModel):
    kb_id: str
    url: str
    title: str | None = None


@router.post("/import-url", response_model=DocResponse)
async def import_url_document(body: ImportUrlBody, session: Session = Depends(get_session)):
    import httpx as _httpx

    kb = session.get(KnowledgeBase, body.kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    if not body.url.startswith(("http://", "https://")):
        raise HTTPException(422, "Invalid URL")

    try:
        async with _httpx.AsyncClient(timeout=30, follow_redirects=True) as client_:
            resp = await client_.get(body.url)
            resp.raise_for_status()
            data = resp.content
    except Exception as e:
        raise HTTPException(502, f"URL fetch failed: {e}")

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Max {settings.max_upload_size_mb}MB")

    filename = body.title or body.url.rstrip("/").rsplit("/", 1)[-1] or "page"
    if "html" in resp.headers.get("content-type", "") and not filename.lower().endswith((".html", ".htm")):
        filename += ".html"

    file_hash = hashlib.sha256(data).hexdigest()
    existing = (
        session.query(Knowledge)
        .filter(Knowledge.knowledge_base_id == body.kb_id, Knowledge.file_hash == file_hash,
                Knowledge.deleted_at.is_(None))
        .first()
    )
    if existing:
        return existing

    doc_id = uuid.uuid4().hex
    original_key = f"originals/{body.kb_id}/{doc_id}/{filename}"
    doc = Knowledge(
        id=doc_id, tenant_id=kb.tenant_id, knowledge_base_id=body.kb_id, type="file",
        title=filename, file_name=filename, file_type=resp.headers.get("content-type", ""),
        file_size=len(data), file_hash=file_hash, source="url", parse_status="processing",
        uploaded_by="admin",
        metadata_={"original_key": original_key, "source_url": body.url},
    )
    session.add(doc)
    session.commit()
    await _save_and_dispatch(doc_id, body.kb_id, kb.tenant_id, data, filename, original_key)
    return doc


@router.get("")
def list_documents(
    kb_id: str,
    page: int = 1,
    size: int = 20,
    keyword: str | None = None,
    status: str | None = None,
    tag: str | None = None,
    file_type: str | None = None,
    sort_by: str = "created_at",
    order: str = "desc",
    session: Session = Depends(get_session),
):
    from sqlalchemy import cast, String as SAString

    query = (
        session.query(Knowledge)
        .filter(Knowledge.knowledge_base_id == kb_id, Knowledge.deleted_at.is_(None))
    )
    if keyword:
        pattern = f"%{keyword}%"
        query = query.filter(Knowledge.title.ilike(pattern) | Knowledge.file_name.ilike(pattern))
    if status:
        query = query.filter(Knowledge.parse_status == status)
    if file_type:
        query = query.filter(Knowledge.file_type.ilike(f"%{file_type}%"))
    if tag:
        import json as _json

        # SQLAlchemy JSON 默认 ensure_ascii，中文标签落库为 \uXXXX 转义；两种形态都匹配
        escaped = _json.dumps(tag)[1:-1]
        col = cast(Knowledge.tags, SAString)
        cond = col.like(f'%"{tag}"%')
        if escaped != tag:
            cond = cond | col.like(f'%"{escaped}"%')
        query = query.filter(cond)

    sort_col = {
        "created_at": Knowledge.created_at,
        "updated_at": Knowledge.updated_at,
        "title": Knowledge.title,
        "file_size": Knowledge.file_size,
    }.get(sort_by, Knowledge.created_at)
    query = query.order_by(sort_col.asc() if order == "asc" else sort_col.desc())

    total = query.count()
    items = query.offset((page - 1) * size).limit(size).all()
    return {
        "items": [DocResponse.model_validate(d) for d in items],
        "total": total,
        "page": page,
        "size": size,
    }


@router.get("/status/stream")
async def stream_document_statuses(kb_id: str):
    """SSE 状态流：替代前端逐文档轮询，状态变化才推送，解析全部结束后关流。"""
    import asyncio
    import json

    from sse_starlette.sse import EventSourceResponse

    from ...models.database import SessionLocal

    def _snapshot() -> list[dict]:
        session = SessionLocal()
        try:
            docs = (
                session.query(Knowledge)
                .filter(Knowledge.knowledge_base_id == kb_id, Knowledge.deleted_at.is_(None))
                .all()
            )
            return [
                {
                    "doc_id": d.id,
                    "parse_status": d.parse_status,
                    "graph_status": d.graph_status,
                    "wiki_status": d.wiki_status,
                    "error_message": d.error_message,
                }
                for d in docs
            ]
        finally:
            session.close()

    async def event_gen():
        last: list[dict] | None = None
        while True:
            snap = await asyncio.to_thread(_snapshot)
            if snap != last:
                yield {"event": "status", "data": json.dumps(snap, ensure_ascii=False)}
                last = snap
            if not any(s["parse_status"] == "processing" for s in snap):
                yield {"event": "done", "data": "{}"}
                return
            await asyncio.sleep(1.0)

    return EventSourceResponse(event_gen())


@router.get("/{doc_id}", response_model=DocResponse)
def get_document(doc_id: str, session: Session = Depends(get_session)):
    doc = session.get(Knowledge, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


@router.get("/{doc_id}/status")
def get_document_status(doc_id: str, session: Session = Depends(get_session)):
    from ...tasks.parse_tasks import get_progress

    doc = session.get(Knowledge, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    progress = get_progress(doc_id) or {
        "stage": doc.parse_status,
        "percent": 100 if doc.parse_status == "completed" else 0,
    }

    return {
        "doc_id": doc_id,
        "parse_status": doc.parse_status,
        "graph_status": doc.graph_status,
        "wiki_status": doc.wiki_status,
        "progress": progress,
        "error_message": doc.error_message,
    }


class DocUpdate(BaseModel):
    title: str | None = None
    tags: list[str] | None = None
    authority: int | None = None
    uploaded_by: str | None = None
    wiki_enabled: bool | None = None


@router.patch("/{doc_id}", response_model=DocResponse)
def update_document(doc_id: str, body: DocUpdate, session: Session = Depends(get_session)):
    doc = session.get(Knowledge, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    if body.title is not None:
        name = body.title.strip()
        if not name:
            raise HTTPException(422, "名称不能为空")
        doc.title = name
        doc.file_name = name
    if body.tags is not None:
        doc.tags = [t.strip() for t in body.tags if t.strip()]
    if body.authority is not None:
        if body.authority not in (0, 1):
            raise HTTPException(422, "authority 仅支持 0/1")
        doc.authority = body.authority
    if body.uploaded_by is not None:
        doc.uploaded_by = body.uploaded_by
    if body.wiki_enabled is not None:
        doc.metadata_ = {**(doc.metadata_ or {}), "wiki_enabled": body.wiki_enabled}
    session.commit()
    session.refresh(doc)
    return doc


@router.get("/{doc_id}/download")
async def download_document(doc_id: str, session: Session = Depends(get_session)):
    from urllib.parse import quote

    from fastapi.responses import Response

    from ...storage.factory import get_storage

    doc = session.get(Knowledge, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    meta = doc.metadata_ or {}
    key = meta.get("original_key")
    if not key:
        raise HTTPException(404, "原文件未存档（旧版本上传的文档），请重新上传")
    data = await get_storage().objects.get(key)
    if data is None:
        raise HTTPException(404, "原文件已丢失")
    filename = meta.get("original_filename") or doc.file_name or "file"
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
    )


@router.post("/{doc_id}/reparse")
async def reparse_document(doc_id: str, session: Session = Depends(get_session)):
    from ...fastpath.cache import mark_entries_stale_for_document
    from ...storage.factory import get_storage

    doc = session.get(Knowledge, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    kb = session.get(KnowledgeBase, doc.knowledge_base_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    meta = doc.metadata_ or {}
    key = meta.get("original_key")
    if not key:
        raise HTTPException(409, "原文件未存档（旧版本上传的文档），请删除后重新上传")
    data = await get_storage().objects.get(key)
    if data is None:
        raise HTTPException(409, "原文件已丢失，请删除后重新上传")

    doc.parse_status = "processing"
    doc.error_message = None
    session.commit()

    mark_entries_stale_for_document(session, doc_id)
    filename = meta.get("original_filename") or doc.file_name or "file"
    _dispatch_processing(kb, doc.id, data, filename)
    log.info("重新解析已触发", doc_id=doc.id, filename=filename)
    return {"status": "processing"}


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, session: Session = Depends(get_session)):
    from datetime import timezone

    from sqlalchemy import select, update

    from ...graph.builder import get_graph_manager
    from ...models.chunk import Chunk
    from ...storage.factory import get_storage

    doc = session.get(Knowledge, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    now = datetime.now(timezone.utc)
    doc.deleted_at = now
    live_chunk_ids = list(
        session.execute(
            select(Chunk.id).where(Chunk.knowledge_id == doc_id, Chunk.deleted_at.is_(None))
        ).scalars()
    )
    session.execute(
        update(Chunk)
        .where(Chunk.knowledge_id == doc_id, Chunk.deleted_at.is_(None))
        .values(deleted_at=now)
    )
    session.commit()

    from ...fastpath.cache import (
        invalidate_cache_for_chunks,
        mark_entries_stale_for_chunks,
    )

    mark_entries_stale_for_chunks(session, live_chunk_ids)
    invalidate_cache_for_chunks(session, live_chunk_ids)

    kb_id = doc.knowledge_base_id
    storage = get_storage()
    await storage.vector.delete_document(kb_id, doc_id)
    await storage.keyword.delete_document(kb_id, doc_id)
    from ...indexing.reconcile import live_text_chunk_ids

    await get_graph_manager().reconcile(kb_id, live_text_chunk_ids(session, kb_id))
    log.info("文档删除_索引已清理", kb_id=kb_id, doc_id=doc_id)

    kb = session.get(KnowledgeBase, kb_id)
    if kb and (kb.wiki_config or {}).get("enabled", True):
        import threading

        def _wiki_retract():
            from ...models.database import SessionLocal
            from ...wiki.build import run_wiki_build
            from ...wiki.queue import enqueue_ingest

            s = SessionLocal()
            try:
                enqueue_ingest(s, kb_id, doc_id, op="retract")
                run_wiki_build(s, kb_id)
            except Exception as e:
                log.error("Wiki撤回构建失败", kb_id=kb_id, doc_id=doc_id, error=str(e)[:200])
            finally:
                s.close()

        threading.Thread(target=_wiki_retract, daemon=True).start()

    return {"status": "deleted"}


@router.get("/{doc_id}/chunks")
def list_chunks(
    doc_id: str,
    page: int = 1,
    size: int = 20,
    chunk_type: str | None = None,
    session: Session = Depends(get_session),
):
    from sqlalchemy import case, func
    from sqlalchemy.orm import aliased
    from ...models.chunk import Chunk

    doc = session.get(Knowledge, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    Parent = aliased(Chunk)
    group_start = func.coalesce(Parent.start_at, Chunk.start_at, 0)
    is_parent = case((Chunk.parent_chunk_id.is_(None), 0), else_=1)

    query = (
        session.query(Chunk)
        .outerjoin(Parent, Chunk.parent_chunk_id == Parent.id)
        .filter(Chunk.knowledge_id == doc_id, Chunk.deleted_at.is_(None))
    )
    if chunk_type:
        query = query.filter(Chunk.chunk_type == chunk_type)

    total = query.count()
    items = (
        query.order_by(group_start, is_parent, Chunk.start_at)
        .offset((page - 1) * size)
        .limit(size)
        .all()
    )

    # 子块 start_at/end_at 是相对父块的偏移，转换为文档绝对偏移以配合 /content 坐标系
    parent_ids = {c.parent_chunk_id for c in items if c.parent_chunk_id}
    parent_starts: dict[str, int] = {}
    if parent_ids:
        parents = session.query(Chunk.id, Chunk.start_at).filter(Chunk.id.in_(parent_ids)).all()
        parent_starts = {pid: (s or 0) for pid, s in parents}

    result = []
    for c in items:
        r = ChunkResponse.model_validate(c)
        if c.parent_chunk_id and c.parent_chunk_id in parent_starts:
            base = parent_starts[c.parent_chunk_id]
            r.start_at = base + (c.start_at or 0)
            r.end_at = base + (c.end_at or 0)
        result.append(r)

    return {
        "items": result,
        "total": total,
        "page": page,
        "size": size,
    }


class ChunkUpdateBody(BaseModel):
    content: str | None = None
    enabled: bool | None = None


async def _reindex_chunk(chunk) -> None:
    from ...indexing.embedder import build_embed_content, embed_texts
    from ...storage.factory import get_storage

    meta = chunk.metadata_ or {}
    embed_content = build_embed_content(meta.get("doc_title", ""), meta.get("title_path", []), chunk.content)
    storage = get_storage()
    vectors = await embed_texts([embed_content])
    idx_meta = [{"knowledge_id": chunk.knowledge_id, "chunk_type": chunk.chunk_type, "doc_title": meta.get("doc_title", "")}]
    await storage.vector.upsert(chunk.knowledge_base_id, [chunk.id], vectors, idx_meta)
    await storage.keyword.index_chunks(chunk.knowledge_base_id, [chunk.id], [embed_content], idx_meta)


@router.put("/{doc_id}/chunks/{chunk_id}")
async def update_chunk(doc_id: str, chunk_id: str, body: ChunkUpdateBody, session: Session = Depends(get_session)):
    from ...fastpath.cache import (
        invalidate_cache_for_chunks,
        mark_entries_stale_for_chunks,
    )
    from ...indexing.embedder import content_hash
    from ...models.chunk import Chunk
    from ...storage.factory import get_storage

    chunk = session.get(Chunk, chunk_id)
    if not chunk or chunk.knowledge_id != doc_id or chunk.deleted_at is not None:
        raise HTTPException(404, "Chunk not found")

    if body.content is not None:
        chunk.content = body.content
        chunk.content_hash = content_hash(body.content)

    toggled = body.enabled is not None and body.enabled != chunk.enabled
    if body.enabled is not None:
        chunk.enabled = body.enabled
    session.commit()

    if body.content is not None:
        mark_entries_stale_for_chunks(session, [chunk_id])
        invalidate_cache_for_chunks(session, [chunk_id])

    if chunk.chunk_type == "parent_text":
        return {"status": "updated", "enabled": chunk.enabled}

    if not chunk.enabled:
        if toggled:
            storage = get_storage()
            await storage.vector.delete_chunks(chunk.knowledge_base_id, [chunk_id])
            await storage.keyword.delete_chunks(chunk.knowledge_base_id, [chunk_id])
            log.info("切块禁用_索引已移除", chunk_id=chunk_id, doc_id=doc_id)
        return {"status": "updated", "enabled": False}

    if body.content is not None or toggled:
        await _reindex_chunk(chunk)
    log.info("切块更新", chunk_id=chunk_id, doc_id=doc_id, enabled=chunk.enabled)
    return {"status": "updated", "enabled": chunk.enabled}


@router.delete("/{doc_id}/chunks/{chunk_id}")
async def delete_chunk(doc_id: str, chunk_id: str, session: Session = Depends(get_session)):
    from datetime import datetime, timezone

    from ...fastpath.cache import (
        invalidate_cache_for_chunks,
        mark_entries_stale_for_chunks,
    )
    from ...models.chunk import Chunk
    from ...storage.factory import get_storage

    chunk = session.get(Chunk, chunk_id)
    if not chunk or chunk.knowledge_id != doc_id or chunk.deleted_at is not None:
        raise HTTPException(404, "Chunk not found")
    chunk.deleted_at = datetime.now(timezone.utc)
    session.commit()
    mark_entries_stale_for_chunks(session, [chunk_id])
    invalidate_cache_for_chunks(session, [chunk_id])
    storage = get_storage()
    await storage.vector.delete_chunks(chunk.knowledge_base_id, [chunk_id])
    await storage.keyword.delete_chunks(chunk.knowledge_base_id, [chunk_id])
    log.info("切块删除", chunk_id=chunk_id, doc_id=doc_id)
    return {"status": "deleted"}


@router.get("/{doc_id}/content")
def get_document_content(doc_id: str, session: Session = Depends(get_session)):
    from ...models.chunk import Chunk

    doc = session.get(Knowledge, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")

    def _load(chunk_type: str):
        return (
            session.query(Chunk)
            .filter(
                Chunk.knowledge_id == doc_id,
                Chunk.chunk_type == chunk_type,
                Chunk.deleted_at.is_(None),
                Chunk.start_at.is_not(None),
            )
            .order_by(Chunk.start_at)
            .all()
        )

    chunks = _load("parent_text") or _load("text")
    if not chunks:
        return {"doc_id": doc_id, "content": "", "length": 0}

    # 用切块偏移重建解析后文本，保持与 chunk.start_at/end_at 同一坐标系
    end_max = max(c.end_at or 0 for c in chunks)
    buf = ["\n"] * end_max
    for c in chunks:
        text = c.content[: (c.end_at or 0) - (c.start_at or 0)]
        buf[c.start_at : c.start_at + len(text)] = list(text)
    content = "".join(buf)
    return {"doc_id": doc_id, "content": content, "length": len(content)}
