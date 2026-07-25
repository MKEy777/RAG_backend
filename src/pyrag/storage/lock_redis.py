from __future__ import annotations

import redis.asyncio as aioredis

from ..settings import settings


class RedisLock:
    def __init__(self, url: str | None = None):
        self._url = url or settings.redis_url
        self._client: aioredis.Redis | None = None

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(self._url, decode_responses=False)
        return self._client

    async def acquire(self, key: str, ttl_s: int = 30) -> bool:
        client = await self._get_client()
        return bool(await client.set(f"lock:{key}", b"1", nx=True, ex=ttl_s))

    async def release(self, key: str) -> None:
        client = await self._get_client()
        await client.delete(f"lock:{key}")

    async def get(self, key: str) -> bytes | None:
        client = await self._get_client()
        return await client.get(key)

    async def set(self, key: str, value: bytes, ttl_s: int | None = None) -> None:
        client = await self._get_client()
        await client.set(key, value, ex=ttl_s)
