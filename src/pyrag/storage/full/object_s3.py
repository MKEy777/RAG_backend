from __future__ import annotations


class S3ObjectStore:
    """S3-compatible object store (MinIO / COS / TOS). Placeholder."""

    async def put(self, key: str, data: bytes) -> str:
        raise NotImplementedError("Implement S3ObjectStore.put")

    async def get(self, key: str) -> bytes | None:
        raise NotImplementedError("Implement S3ObjectStore.get")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("Implement S3ObjectStore.delete")

    async def url(self, key: str) -> str:
        raise NotImplementedError("Implement S3ObjectStore.url")
