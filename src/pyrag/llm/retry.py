from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from ..settings import settings


def embed_retry():
    return retry(
        stop=stop_after_attempt(settings.embed_retry_count),
        wait=wait_exponential(multiplier=settings.embed_retry_base_ms / 1000, max=10),
        reraise=True,
    )


def wiki_llm_retry():
    return retry(
        stop=stop_after_attempt(settings.wiki_llm_retry_count),
        wait=wait_exponential(multiplier=settings.wiki_llm_retry_base_s, max=30),
        reraise=True,
    )


def graph_llm_retry():
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2.0, max=20),
        reraise=True,
    )
