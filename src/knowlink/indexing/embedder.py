from __future__ import annotations

import asyncio
import hashlib

import numpy as np

from ..logging import get_logger, Timer
from ..llm.client import get_llm_client
from ..llm.retry import embed_retry
from ..settings import settings

log = get_logger("embedder")


def build_embed_content(doc_title: str, title_path: list[str], content: str) -> str:
    prefix_parts = []
    if doc_title:
        prefix_parts.append(doc_title)
    if title_path:
        prefix_parts.extend(title_path)
    prefix = " > ".join(prefix_parts)
    if prefix:
        text = f"{prefix}\n{content}"
    else:
        text = content
    if len(text) > settings.embed_max_chars:
        text = text[: settings.embed_max_chars]
    return text


async def embed_texts(texts: list[str], model: str | None = None) -> np.ndarray:
    client = get_llm_client()
    all_vectors: list[list[float]] = []
    batch_size = settings.batch_embed_size
    total = len(texts)
    log.info("嵌入开始", total_chunks=total, batch_size=batch_size)

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]

        @embed_retry()
        async def _call(b: list[str]) -> list[list[float]]:
            return await client.embed(b, model=model)

        vectors = await _call(batch)
        all_vectors.extend(vectors)

    arr = np.array(all_vectors, dtype=np.float32)
    log.info("嵌入完成", total=total, shape=list(arr.shape))
    return arr


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
