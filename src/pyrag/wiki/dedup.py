from __future__ import annotations

from ..llm.client import get_llm_client
from ..llm.json_repair import parse_json_object
from ..llm.retry import wiki_llm_retry
from ..models.wiki import WikiPage
from ..prompts.loader import get_prompt

DEDUP_TOP_K = 5           # detail.md §Wiki 去重与归并
DEDUP_SCORE_FLOOR = 0.08  # detail.md §Wiki 去重与归并
DEDUP_SMALL_CORPUS = 25   # detail.md §Wiki 去重与归并


def _bigrams(text: str) -> set[str]:
    t = (text or "").lower().replace(" ", "")
    if len(t) < 2:
        return {t} if t else set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def surface_score(candidate: dict, page: WikiPage) -> float:
    names = [candidate.get("name", "")] + list(candidate.get("aliases") or [])
    targets = [page.title] + list(page.aliases or [])
    best = 0.0
    for n in names:
        ng = _bigrams(n)
        for t in targets:
            best = max(best, _jaccard(ng, _bigrams(t)))
    return best


def prefilter_candidates(candidate: dict, pages: list[WikiPage]) -> list[WikiPage]:
    scored = [(surface_score(candidate, p), p) for p in pages]
    scored = [(s, p) for s, p in scored if s >= DEDUP_SCORE_FLOOR]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:DEDUP_TOP_K]]


async def dedup_candidates(session, kb_id: str, candidates: list[dict]) -> dict[str, str]:
    """Return {new_slug: existing_slug}; only merges whose target page really exists."""
    existing = (
        session.query(WikiPage)
        .filter(
            WikiPage.knowledge_base_id == kb_id,
            WikiPage.kind.in_(["entity", "concept"]),
            WikiPage.deleted_at.is_(None),
        )
        .all()
    )
    if not existing:
        return {}

    existing_by_slug = {p.slug: p for p in existing}
    new_items = [c for c in candidates if c.get("slug") and c["slug"] not in existing_by_slug]
    if not new_items:
        return {}

    small_corpus = len(existing) <= DEDUP_SMALL_CORPUS
    items_desc: list[str] = []
    for c in new_items:
        pool = existing if small_corpus else prefilter_candidates(c, existing)
        if not pool:
            continue
        cand_lines = "\n".join(
            f"  - slug={p.slug} title={p.title} aliases={list(p.aliases or [])} type={p.kind}"
            for p in pool
        )
        items_desc.append(
            f'- new_slug={c["slug"]} name={c.get("name", "")} aliases={c.get("aliases") or []} '
            f'type={c.get("_type", "entity")}\n<candidates>\n{cand_lines}\n</candidates>'
        )
    if not items_desc:
        return {}

    client = get_llm_client()
    user_msg = "<items>\n" + "\n".join(items_desc) + "\n</items>"

    @wiki_llm_retry()
    async def _call():
        return await client.chat(
            [
                {"role": "system", "content": get_prompt("wiki", "WikiDeduplicationPrompt")},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=2048,
            purpose="wiki:dedup",
        )

    try:
        data = parse_json_object(await _call())
    except Exception:
        return {}

    merges = data.get("merges") or {}
    # 只接受目标页真实存在的合并，防 LLM 幻觉
    return {ns: es for ns, es in merges.items() if es in existing_by_slug and ns != es}
