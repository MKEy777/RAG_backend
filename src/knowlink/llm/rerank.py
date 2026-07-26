from __future__ import annotations

import httpx

from ..settings import settings


class Reranker:
    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self._base_url = (base_url or settings.llm_rerank_base_url).rstrip("/")
        self._api_key = api_key or settings.llm_api_key
        self._model = settings.llm_rerank_model
        self._client = httpx.AsyncClient(timeout=60.0)

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[dict]:
        top_k = top_k or settings.rerank_top_k
        resp = await self._client.post(
            self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "query": query,
                "documents": documents,
                "top_n": top_k,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("results", []):
            results.append({
                "index": item["index"],
                "score": item["relevance_score"],
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    async def close(self) -> None:
        await self._client.aclose()


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
