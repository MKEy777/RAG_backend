from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..logging import get_logger
from ..settings import settings

log = get_logger("vector_store")
MAX_RESIDENT_KB = 32


class LocalVectorStore:
    def __init__(self, root: Path | None = None):
        self._root = root or settings.object_storage_root / "vectors"
        self._root.mkdir(parents=True, exist_ok=True)
        self._indices: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._meta: dict[str, dict[str, dict]] = {}

    def _evict_if_needed(self) -> None:
        while len(self._indices) > MAX_RESIDENT_KB:
            evicted_kb, _ = self._indices.popitem(last=False)
            self._meta.pop(evicted_kb, None)

    def _kb_dir(self, kb_id: str) -> Path:
        d = self._root / kb_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_kb(self, kb_id: str) -> None:
        if kb_id in self._indices:
            self._indices.move_to_end(kb_id)
            return
        kb_dir = self._kb_dir(kb_id)
        vec_file = kb_dir / "vectors.npy"
        ids_file = kb_dir / "ids.npy"
        if vec_file.exists() and ids_file.exists():
            vectors = np.load(str(vec_file)).astype(np.float32)
            ids = np.load(str(ids_file), allow_pickle=True).tolist()
            self._indices[kb_id] = {cid: vectors[i] for i, cid in enumerate(ids)}
        else:
            self._indices[kb_id] = {}
        meta_file = kb_dir / "meta.json"
        if meta_file.exists():
            self._meta[kb_id] = json.loads(meta_file.read_text(encoding="utf-8"))
        else:
            self._meta.setdefault(kb_id, {})
        self._evict_if_needed()

    def _save_kb(self, kb_id: str) -> None:
        kb_dir = self._kb_dir(kb_id)
        index = self._indices.get(kb_id, {})
        if not index:
            for name in ("vectors.npy", "ids.npy", "meta.json"):
                f = kb_dir / name
                if f.exists():
                    f.unlink()
            return
        ids = list(index.keys())
        vectors = np.array([index[cid] for cid in ids], dtype=np.float16)
        np.save(str(kb_dir / "vectors.npy"), vectors)
        np.save(str(kb_dir / "ids.npy"), np.array(ids, dtype=object))
        (kb_dir / "meta.json").write_text(
            json.dumps(self._meta.get(kb_id, {}), ensure_ascii=False), encoding="utf-8"
        )

    async def upsert(
        self, kb_id: str, chunk_ids: list[str], vectors: np.ndarray, metadata: list[dict]
    ) -> None:
        def _do():
            self._load_kb(kb_id)
            for i, cid in enumerate(chunk_ids):
                self._indices[kb_id][cid] = vectors[i].astype(np.float32)
                if i < len(metadata):
                    self._meta[kb_id][cid] = metadata[i]
            self._save_kb(kb_id)
            log.info("向量写入", kb_id=kb_id, count=len(chunk_ids))

        await asyncio.to_thread(_do)

    async def search(
        self, kb_id: str, query_vector: np.ndarray, top_k: int, threshold: float = 0.0
    ) -> list[dict]:
        def _do() -> list[dict]:
            self._load_kb(kb_id)
            index = self._indices.get(kb_id, {})
            if not index:
                return []
            ids = list(index.keys())
            mat = np.array([index[cid] for cid in ids], dtype=np.float32)
            q = query_vector.astype(np.float32)
            q_norm = q / (np.linalg.norm(q) + 1e-10)
            mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10)
            scores = mat_norm @ q_norm
            results = []
            for i, score in enumerate(scores):
                s = float(score)
                if s >= threshold:
                    results.append({
                        "chunk_id": ids[i],
                        "score": s,
                        "metadata": self._meta.get(kb_id, {}).get(ids[i], {}),
                    })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        return await asyncio.to_thread(_do)

    async def delete_chunks(self, kb_id: str, chunk_ids: list[str]) -> None:
        def _do():
            self._load_kb(kb_id)
            for cid in chunk_ids:
                self._indices[kb_id].pop(cid, None)
                self._meta.get(kb_id, {}).pop(cid, None)
            self._save_kb(kb_id)

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
                self._indices[kb_id].pop(cid, None)
                self._meta.get(kb_id, {}).pop(cid, None)
            self._save_kb(kb_id)
            log.info("向量删除文档", kb_id=kb_id, knowledge_id=knowledge_id, removed=len(to_remove))

        await asyncio.to_thread(_do)

    async def list_ids(self, kb_id: str) -> list[str]:
        """列出 kb 命名空间下全部向量 id（管理运维用）。"""

        def _do() -> list[str]:
            self._load_kb(kb_id)
            return list(self._indices.get(kb_id, {}).keys())

        return await asyncio.to_thread(_do)

    async def reconcile(self, kb_id: str, live_meta: dict[str, dict]) -> int:
        """以 DB 为准对账：移除死块向量，回填历史缺失的 metadata（meta 早期未落盘）。"""
        def _do() -> int:
            self._load_kb(kb_id)
            index = self._indices.get(kb_id, {})
            meta = self._meta.setdefault(kb_id, {})
            removed = [cid for cid in index if cid not in live_meta]
            for cid in removed:
                index.pop(cid, None)
                meta.pop(cid, None)
            backfilled = 0
            for cid in index:
                if not meta.get(cid):
                    meta[cid] = live_meta[cid]
                    backfilled += 1
            if removed or backfilled:
                self._save_kb(kb_id)
                log.info("向量对账", kb_id=kb_id, removed=len(removed), backfilled=backfilled)
            return len(removed)

        return await asyncio.to_thread(_do)
