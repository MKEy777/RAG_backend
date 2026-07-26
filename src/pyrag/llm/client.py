from __future__ import annotations

import json
import time
from typing import AsyncIterator

import httpx

from ..logging import get_logger
from ..settings import settings
from .circuit_breaker import CircuitBreaker, CircuitOpenError

log = get_logger("llm")

_chat_breaker = CircuitBreaker("llm_chat", failure_threshold=5, recovery_timeout_s=30)
_embed_breaker = CircuitBreaker("llm_embed", failure_threshold=5, recovery_timeout_s=30)

TIMEOUT_EMBED = 30.0
TIMEOUT_CHAT = 120.0
TIMEOUT_VLM = 60.0


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Ensure messages are in OpenAI chat completions format."""
    result = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if item.get("type") == "text":
                    parts.append({"type": "text", "text": item["text"]})
                elif item.get("type") == "image_url":
                    parts.append({"type": "image_url", "image_url": item["image_url"]})
                elif item.get("type") == "image":
                    src = item.get("source", {})
                    if src.get("type") == "base64":
                        data_url = f"data:{src.get('media_type', 'image/jpeg')};base64,{src['data']}"
                        parts.append({"type": "image_url", "image_url": {"url": data_url}})
            content = parts
        result.append({"role": msg["role"], "content": content})
    return result


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self._base_url = (base_url or settings.llm_base_url).rstrip("/")
        self._api_key = api_key or settings.llm_api_key
        self._embed_base_url = settings.llm_embed_base_url.rstrip("/")
        self._embed_api_key = settings.llm_embed_api_key

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        purpose: str = "",
    ) -> str:
        if not _chat_breaker.allow_request():
            raise CircuitOpenError("llm_chat")
        model = model or settings.llm_chat_model
        body: dict = {
            "model": model,
            "messages": _normalize_messages(messages),
            "max_tokens": max_tokens,
            "temperature": max(temperature, 0.01),
        }

        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_CHAT) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=body,
                )
            if resp.status_code >= 400:
                detail = resp.text[:500]
                log.error("LLM请求被拒绝", status=resp.status_code, detail=detail, model=model)
                resp.raise_for_status()
            data = resp.json()
            _chat_breaker.record_success()
            log.info("LLM对话成功", purpose=purpose, model=model, duration_ms=round((time.perf_counter() - start) * 1000, 1))
            return data["choices"][0]["message"]["content"]
        except CircuitOpenError:
            raise
        except Exception as e:
            _chat_breaker.record_failure()
            log.error("LLM对话失败", purpose=purpose, model=model, error=str(e), duration_ms=round((time.perf_counter() - start) * 1000, 1))
            raise

    async def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        model = model or settings.llm_chat_model
        body: dict = {
            "model": model,
            "messages": _normalize_messages(messages),
            "max_tokens": max_tokens,
            "temperature": max(temperature, 0.01),
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=TIMEOUT_CHAT) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=body,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = event.get("choices", [])
                    if choices:
                        text = choices[0].get("delta", {}).get("content", "")
                        if text:
                            yield text

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """Embedding uses OpenAI-compatible format via SiliconFlow."""
        if not _embed_breaker.allow_request():
            raise CircuitOpenError("llm_embed")
        model = model or settings.llm_embedding_model
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_EMBED) as client:
                resp = await client.post(
                    f"{self._embed_base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self._embed_api_key}"},
                    json={"model": model, "input": texts},
                )
            resp.raise_for_status()
            data = resp.json()
            _embed_breaker.record_success()
            return [item["embedding"] for item in data["data"]]
        except CircuitOpenError:
            raise
        except Exception as e:
            _embed_breaker.record_failure()
            log.error("LLM嵌入失败", model=model, error=str(e), duration_ms=round((time.perf_counter() - start) * 1000, 1))
            raise

    async def vlm(
        self,
        image_base64: str,
        prompt: str,
        model: str | None = None,
        mime: str = "image/jpeg",
        purpose: str = "",
    ) -> str:
        if not _chat_breaker.allow_request():
            raise CircuitOpenError("llm_chat")
        model = model or settings.llm_vlm_model
        data_url = f"data:{mime};base64,{image_base64}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_VLM) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json={"model": model, "messages": messages, "max_tokens": 4096},
                )
            if resp.status_code >= 400:
                log.error("VLM请求被拒绝", status=resp.status_code, detail=resp.text[:500], model=model)
                resp.raise_for_status()
            data = resp.json()
            _chat_breaker.record_success()
            log.info("LLM视觉模型成功", purpose=purpose, model=model, duration_ms=round((time.perf_counter() - start) * 1000, 1))
            return data["choices"][0]["message"]["content"]
        except CircuitOpenError:
            raise
        except Exception as e:
            _chat_breaker.record_failure()
            log.error("LLM视觉模型失败", purpose=purpose, model=model, error=str(e), duration_ms=round((time.perf_counter() - start) * 1000, 1))
            raise


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
