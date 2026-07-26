from __future__ import annotations

import asyncio
import threading


class LocalLock:
    def __init__(self):
        self._locks: dict[str, threading.Lock] = {}
        self._store: dict[str, bytes] = {}
        self._mu = threading.Lock()

    def _get_lock(self, key: str) -> threading.Lock:
        with self._mu:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    async def acquire(self, key: str, ttl_s: int = 30) -> bool:
        lock = self._get_lock(key)
        acquired = lock.acquire(blocking=False)
        return acquired

    async def release(self, key: str) -> None:
        lock = self._get_lock(key)
        try:
            lock.release()
        except RuntimeError:
            pass

    async def get(self, key: str) -> bytes | None:
        return self._store.get(key)

    async def set(self, key: str, value: bytes, ttl_s: int | None = None) -> None:
        self._store[key] = value
