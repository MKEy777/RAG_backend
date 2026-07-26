from __future__ import annotations

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from ..settings import settings
from .lock_local import LocalLock


class RedisLock:
    def __init__(self, url: str | None = None):
        self._url = url or settings.redis_url
        self._client: aioredis.Redis | None = None
        # Redis 不可用时降级为进程内锁，避免同步路径被 ConnectionError 中断
        self._fallback = LocalLock()

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(self._url, decode_responses=False)
        return self._client

    async def acquire(self, key: str, ttl_s: int = 30) -> bool:
        try:
            client = await self._get_client()
            return bool(await client.set(f"lock:{key}", b"1", nx=True, ex=ttl_s))
        except (RedisError, OSError):
            return await self._fallback.acquire(key, ttl_s)

    async def release(self, key: str) -> None:
        try:
            client = await self._get_client()
            await client.delete(f"lock:{key}")
        except (RedisError, OSError):
            await self._fallback.release(key)

    async def get(self, key: str) -> bytes | None:
        try:
            client = await self._get_client()
            return await client.get(key)
        except (RedisError, OSError):
            return await self._fallback.get(key)

    async def set(self, key: str, value: bytes, ttl_s: int | None = None) -> None:
        try:
            client = await self._get_client()
            await client.set(key, value, ex=ttl_s)
        except (RedisError, OSError):
            await self._fallback.set(key, value, ttl_s)
