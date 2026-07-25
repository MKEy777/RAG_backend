from __future__ import annotations

import asyncio
from pathlib import Path

import bm25s
import jieba

from ..logging import get_logger
from ..settings import settings

log = get_logger("keyword_store")


def _tokenize(text: str) -> list[str]:
    return [w for w in jieba.cut(text) if w.strip()]


class BM25KeywordIndex:
    def __init__(self, root: Path | None = None):
        self._root = root or settings.object_storage_root / "bm25"
        self._root.mkdir(parents=True, exist_ok=True)
        self._corpus: dict[str, dict[str, str]] = {}
        self._meta: dict[str, dict[str, dict]] = {}
        self._retriever: dict[str, bm25s.BM25] = {}
        self._dirty: set[str] = set()

    def _kb_dir(self, kb_id: str) -> Path:
        d = self._root / kb_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_kb(self, kb_id: str) -> None:
        if kb_id in self._corpus:
            return
        kb_dir = self._kb_dir(kb_id)
        index_dir = kb_dir / "index"
        if index_dir.exists():
            try:
                retriever = bm25s.BM25.load(str(index_dir), load_corpus=False)
                self._retriever[kb_id] = retriever
            except Exception:
                pass
        self._corpus.setdefault(kb_id, {})
        self._meta.setdefault(kb_id, {})

    def _mark_dirty(self, kb_id: str) -> None:
        self._dirty.add(kb_id)

    def _rebuild(self, kb_id: str) -> None:
        corpus = self._corpus.get(kb_id, {})
        if not corpus:
            self._retriever.pop(kb_id, None)
            return
        ids = list(corpus.keys())
        texts = [corpus[cid] for cid in ids]
        tokenized = [_tokenize(t) for t in texts]
        retriever = bm25s.BM25()
        retriever.index(tokenized)
        self._retriever[kb_id] = retriever
        kb_dir = self._kb_dir(kb_id)
        retriever.save(str(kb_dir / "index"))
        self._dirty.discard(kb_id)

    async def index_chunks(
        self, kb_id: str, chunk_ids: list[str], texts: list[str], metadata: list[dict]
    ) -> None:
        def _do():
            self._load_kb(kb_id)
            for i, cid in enumerate(chunk_ids):
                self._corpus[kb_id][cid] = texts[i]
                if i < len(metadata):
                    self._meta[kb_id][cid] = metadata[i]
            self._mark_dirty(kb_id)
            log.info("关键词索引写入", kb_id=kb_id, count=len(chunk_ids))

        await asyncio.to_thread(_do)

    async def search(
        self, kb_id: str, query: str, top_k: int, threshold: float = 0.0
    ) -> list[dict]:
        def _do() -> list[dict]:
            self._load_kb(kb_id)
            if kb_id in self._dirty:
                self._rebuild(kb_id)
            retriever = self._retriever.get(kb_id)
            if retriever is None:
                return []
            corpus = self._corpus.get(kb_id, {})
            if not corpus:
                return []
            ids = list(corpus.keys())
            query_tokens = [_tokenize(query)]
            results, scores = retriever.retrieve(query_tokens, k=min(top_k, len(ids)))
            hits = []
            for idx, score in zip(results[0], scores[0]):
                if idx < 0 or idx >= len(ids):
                    continue
                s = float(score)
                if s >= threshold:
                    hits.append({
                        "chunk_id": ids[idx],
                        "score": s,
                        "metadata": self._meta.get(kb_id, {}).get(ids[idx], {}),
                    })
            return hits

        return await asyncio.to_thread(_do)

    async def delete_chunks(self, kb_id: str, chunk_ids: list[str]) -> None:
        def _do():
            self._load_kb(kb_id)
            for cid in chunk_ids:
                self._corpus[kb_id].pop(cid, None)
                self._meta.get(kb_id, {}).pop(cid, None)
            self._mark_dirty(kb_id)
            self._rebuild(kb_id)

        await asyncio.to_thread(_do)

    async def delete_document(self, kb_id: str, knowledge_id: str) -> None:
        def _do():
            self._load_kb(kb_id)
            to_remove = [
                cid
                for cid, meta in self._meta.get(kb_id, {}).items()
                if meta.get("knowledge_id") == knowledge_id
            ]
            for cid in to_remove:
                self._corpus[kb_id].pop(cid, None)
                self._meta.get(kb_id, {}).pop(cid, None)
            self._mark_dirty(kb_id)
            self._rebuild(kb_id)

        await asyncio.to_thread(_do)

    async def rebuild(self, kb_id: str) -> None:
        def _do():
            self._load_kb(kb_id)
            self._rebuild(kb_id)

        await asyncio.to_thread(_do)
