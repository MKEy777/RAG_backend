from __future__ import annotations

import json

from ..llm.client import get_llm_client
from ..logging import get_logger

log = get_logger("rewrite")

REWRITE_PROMPT = (
    "You rewrite the user's latest question into a fully self-contained search query, "
    "resolving pronouns and references using the conversation. Keep the original language. "
    'Return ONLY JSON: {"rewrite_query": "..."}'
)

MAX_HISTORY_ROUNDS = 3


async def rewrite_query(query: str, history: list[dict] | None) -> str:
    if not history:
        return query
    recent = history[-MAX_HISTORY_ROUNDS * 2:]
    convo = "\n".join(f"{m.get('role')}: {str(m.get('content', ''))[:500]}" for m in recent)
    client = get_llm_client()
    try:
        resp = await client.chat(
            [
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": f"<conversation>\n{convo}\n</conversation>\n\nQuestion: {query}"},
            ],
            temperature=0.1,
            max_tokens=150,
            purpose="chat:rewrite",
        )
        cleaned = resp.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        rewritten = str(json.loads(cleaned).get("rewrite_query", "")).strip()
        if rewritten:
            log.info("查询改写", original=query, rewritten=rewritten)
            return rewritten
    except Exception as e:
        log.warning("查询改写失败_使用原查询", error=repr(e))
    return query
