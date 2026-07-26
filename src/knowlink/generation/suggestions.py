from __future__ import annotations

from ..llm.client import get_llm_client
from ..llm.json_repair import parse_json_array
from ..logging import get_logger

log = get_logger("suggestions")

SUGGEST_PROMPT = (
    "Based on the Q&A below, propose 3 short follow-up questions the user is likely to ask next. "
    "Same language as the user. Return ONLY a JSON array of strings."
)


async def generate_suggestions(query: str, answer: str) -> list[str]:
    client = get_llm_client()
    try:
        resp = await client.chat(
            [
                {"role": "system", "content": SUGGEST_PROMPT},
                {"role": "user", "content": f"Q: {query}\nA: {answer[:2000]}"},
            ],
            temperature=0.5,
            max_tokens=200,
            purpose="chat:suggest",
        )
        items = parse_json_array(resp)
        return [str(s).strip() for s in items if str(s).strip()][:3]
    except Exception as e:
        log.warning("建议问题生成失败", error=repr(e))
        return []
