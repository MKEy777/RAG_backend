from __future__ import annotations

import asyncio
import json
import traceback
from datetime import datetime, timezone

from ..logging import get_logger
from ..models.database import SessionLocal
from ..models.knowledge import Knowledge
from ..parsing.base import ParserCapabilityError
from ..parsing.registry import parse_file
from ..settings import settings
from .celery_app import celery_app

log = get_logger("tasks")


def _update_progress(knowledge_id: str, stage: str, percent: int) -> None:
    try:
        import redis as redis_lib

        r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=2)
        r.set(
            f"task_progress:{knowledge_id}",
            json.dumps({"stage": stage, "percent": percent}),
            ex=3600,
        )
    except Exception:
        pass


@celery_app.task(bind=True, max_retries=2)
def parse_and_index_task(self, knowledge_id: str, kb_id: str, tenant_id: int, file_data: bytes, filename: str):
    log.info("解析任务开始", knowledge_id=knowledge_id, filename=filename, size=len(file_data), attempt=self.request.retries)
    session = SessionLocal()
    try:
        _update_progress(knowledge_id, "parsing", 10)
        result = parse_file(file_data, filename)
        log.info("解析任务_文件解析完成", knowledge_id=knowledge_id, markdown_len=len(result.markdown))

        _update_progress(knowledge_id, "chunking", 30)

        from ..indexing.pipeline import run_indexing_pipeline

        kb = session.get(Knowledge, knowledge_id)
        chunking_config = {}
        if kb and kb.metadata_:
            chunking_config = kb.metadata_.get("chunking_config", {})

        _update_progress(knowledge_id, "embedding", 50)
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
        _update_progress(knowledge_id, "completed", 100)
        log.info("解析任务完成", knowledge_id=knowledge_id)
    except ParserCapabilityError as e:
        _update_progress(knowledge_id, "failed", 0)
        session.execute(
            Knowledge.__table__.update()
            .where(Knowledge.id == knowledge_id)
            .values(parse_status="failed", error_message=str(e))
        )
        session.commit()
        log.error("解析任务失败_格式不支持", knowledge_id=knowledge_id, error=str(e))
    except Exception as e:
        _update_progress(knowledge_id, "failed", 0)
        session.execute(
            Knowledge.__table__.update()
            .where(Knowledge.id == knowledge_id)
            .values(parse_status="failed", error_message=traceback.format_exc()[:2000])
        )
        session.commit()
        log.error("解析任务异常_即将重试", knowledge_id=knowledge_id, error=str(e), retries_left=self.max_retries - self.request.retries)
        raise self.retry(exc=e, countdown=5)
    finally:
        session.close()


@celery_app.task(bind=True)
def build_graph_task(self, kb_id: str, knowledge_id: str):
    log.info("图谱构建任务开始", knowledge_id=knowledge_id)
    session = SessionLocal()
    try:
        from sqlalchemy import select

        from ..models.chunk import Chunk
        from ..graph.builder import get_graph_manager

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
        chunk_dicts = [
            {"id": c.id, "content": c.content}
            for c in chunks
        ]
        log.info("图谱构建任务_加载块完成", knowledge_id=knowledge_id, chunk_count=len(chunk_dicts))

        gm = get_graph_manager()
        asyncio.run(gm.build_document(kb_id, knowledge_id, chunk_dicts))
        log.info("图谱构建任务完成", knowledge_id=knowledge_id)
    except Exception as e:
        log.error("图谱构建任务失败", knowledge_id=knowledge_id, error=str(e))
    finally:
        session.close()


@celery_app.task(bind=True)
def wiki_ingest_task(self, kb_id: str, knowledge_id: str):
    pass
