from __future__ import annotations

from dataclasses import dataclass

from ..settings import settings
from .interfaces import CacheLock, KeywordIndex, ObjectStore, VectorStore
from .keyword_bm25s import BM25KeywordIndex
from .lock_local import LocalLock
from .lock_redis import RedisLock
from .object_local import LocalObjectStore
from .vector_local import LocalVectorStore


@dataclass
class StorageBundle:
    vector: VectorStore
    keyword: KeywordIndex
    lock: CacheLock
    objects: ObjectStore


def create_storage(profile: str | None = None) -> StorageBundle:
    profile = profile or settings.storage_profile

    if profile == "full":
        from .full import create_full_storage

        return create_full_storage()

    vector = LocalVectorStore()
    keyword = BM25KeywordIndex()
    objects = LocalObjectStore()

    try:
        lock: CacheLock = RedisLock()
    except Exception:
        lock = LocalLock()

    return StorageBundle(vector=vector, keyword=keyword, lock=lock, objects=objects)


_storage: StorageBundle | None = None


def get_storage() -> StorageBundle:
    global _storage
    if _storage is None:
        _storage = create_storage()
    return _storage
