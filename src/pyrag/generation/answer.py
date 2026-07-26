from __future__ import annotations

import json
from typing import AsyncIterator

from ..llm.client import get_llm_client
from ..retrieval.hybrid import Candidate
from .citations import CITATION_PROTOCOL, build_references
from .context import build_context_xml

SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer ONLY based on the provided sources. "
    "If the sources are insufficient, say 'I cannot answer based on the available knowledge.' "
    "Do not fabricate information. Use inline citations.\n\n"
    f"{CITATION_PROTOCOL}"
)

NO_ANSWER_MSG = "I cannot answer based on the available knowledge."


async def generate_answer_stream(
    query: str,
    candidates: list[Candidate],
    history: list[dict] | None = None,
) -> AsyncIterator[str]:
    if not candidates:
        yield _sse_event("message", {"content": NO_ANSWER_MSG})
        yield _sse_event("references", {"references": []})
        yield _sse_event("done", {})
        return

    context = build_context_xml(candidates)
    references = build_references(candidates)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"<sources>\n{context}\n</sources>\n\nQuestion: {query}",
    })

    client = get_llm_client()
    try:
        async for token in client.chat_stream(messages):
            yield _sse_event("message", {"content": token})
    except Exception:
        yield _sse_event("message", {"content": NO_ANSWER_MSG})

    yield _sse_event("references", {"references": references})
    yield _sse_event("done", {})


async def generate_answer(
    query: str,
    candidates: list[Candidate],
    history: list[dict] | None = None,
) -> dict:
    if not candidates:
        return {"answer": NO_ANSWER_MSG, "references": []}

    context = build_context_xml(candidates)
    references = build_references(candidates)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"<sources>\n{context}\n</sources>\n\nQuestion: {query}",
    })

    client = get_llm_client()
    try:
        answer = await client.chat(messages, purpose="qa:answer")
    except Exception:
        answer = NO_ANSWER_MSG

    return {"answer": answer, "references": references}


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
