from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from ..settings import settings


class LocalObjectStore:
    def __init__(self, root: Path | None = None):
        self._root = root or settings.object_storage_root
        self._root.mkdir(parents=True, exist_ok=True)

    async def put(self, key: str, data: bytes) -> str:
        def _do() -> str:
            path = self._root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return key

        return await asyncio.to_thread(_do)

    async def get(self, key: str) -> bytes | None:
        def _do() -> bytes | None:
            path = self._root / key
            if path.exists():
                return path.read_bytes()
            return None

        return await asyncio.to_thread(_do)

    async def delete(self, key: str) -> None:
        def _do():
            path = self._root / key
            if path.exists():
                path.unlink()

        await asyncio.to_thread(_do)

    async def url(self, key: str) -> str:
        return f"/objects/{key}"
