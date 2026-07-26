from __future__ import annotations

import uuid
from collections import Counter
from itertools import combinations

from sqlalchemy.orm import Session

from ..llm.client import get_llm_client
from ..llm.retry import wiki_llm_retry
from ..models.knowledge_base import KnowledgeBase
from ..models.wiki import WikiPage
from ..prompts.loader import get_prompt

SYNTHESIS_MAX_PAGES = 100      # detail.md §C5 全库正文输入上限
COMPARISON_MIN_COOCCUR = 2     # detail.md §C6 "经常被同时提及"
COMPARISON_MAX_PAIRS = 3       # detail.md §C6


def _split_summary(resp: str) -> tuple[str, str]:
    lines = resp.strip().split("\n")
    if lines and lines[0].startswith("SUMMARY:"):
        return lines[0][8:].strip(), "\n".join(lines[1:]).strip()
    return "", resp.strip()


def _upsert_page(
    session: Session, kb_id: str, tenant_id: int, slug: str, kind: str,
    title: str, content: str, summary: str,
) -> None:
    page = (
        session.query(WikiPage)
        .filter(WikiPage.knowledge_base_id == kb_id, WikiPage.slug == slug)
        .first()
    )
    if page is None:
        session.add(WikiPage(
            page_id=uuid.uuid4().hex, tenant_id=tenant_id, knowledge_base_id=kb_id,
            kind=kind, topic_key=f"{kb_id}:{kind}:{slug}", title=title, slug=slug,
            content=content, summary=summary, status="published", version=1,
        ))
    else:
        page.content = content
        page.summary = summary
        page.deleted_at = None
        page.version = (page.version or 0) + 1
    session.commit()


async def generate_synthesis(session: Session, kb_id: str, tenant_id: int) -> None:
    kb = session.get(KnowledgeBase, kb_id)
    model = ((kb.wiki_config or {}).get("synthesis_model") or "").strip() if kb else ""
    if not model:
        return

    pages = (
        session.query(WikiPage)
        .filter(
            WikiPage.knowledge_base_id == kb_id,
            WikiPage.kind.in_(["overview", "entity", "concept"]),
            WikiPage.status == "published",
            WikiPage.deleted_at.is_(None),
        )
        .all()
    )
    if len(pages) < 2:
        return

    corpus = "\n\n".join(
        f"## [[{p.slug}]] {p.title}\n{p.content or ''}" for p in pages[:SYNTHESIS_MAX_PAGES]
    )
    client = get_llm_client()

    try:
        @wiki_llm_retry()
        async def _call():
            return await client.chat(
                [
                    {"role": "system", "content": get_prompt("wiki", "WikiSynthesisPrompt")},
                    {"role": "user", "content": f"<wiki_pages>\n{corpus}\n</wiki_pages>"},
                ],
                model=model,
                temperature=0.3,
                max_tokens=4096,
                purpose="wiki:synthesis",
            )

        summary, content = _split_summary(await _call())
        if content:
            _upsert_page(session, kb_id, tenant_id, "synthesis/overview", "synthesis",
                         "领域综览", content, summary)
    except Exception:
        pass


def find_comparison_pairs(pages: list[WikiPage]) -> list[tuple[str, str]]:
    """互链共现：实体/概念 slug 对在各页面 out_links 中共同出现的次数。"""
    target_slugs = {p.slug for p in pages if p.kind in ("entity", "concept")}
    counts: Counter = Counter()
    for p in pages:
        linked = sorted((set(p.out_links or []) & target_slugs) - {p.slug})
        for pair in combinations(linked, 2):
            counts[pair] += 1
    pairs = [pair for pair, n in counts.most_common() if n >= COMPARISON_MIN_COOCCUR]
    return pairs[:COMPARISON_MAX_PAIRS]


async def generate_comparisons(session: Session, kb_id: str, tenant_id: int) -> None:
    pages = (
        session.query(WikiPage)
        .filter(
            WikiPage.knowledge_base_id == kb_id,
            WikiPage.status == "published",
            WikiPage.deleted_at.is_(None),
        )
        .all()
    )
    by_slug = {p.slug: p for p in pages}
    pairs = find_comparison_pairs(pages)
    if not pairs:
        return

    client = get_llm_client()
    for slug_a, slug_b in pairs:
        a, b = by_slug.get(slug_a), by_slug.get(slug_b)
        if not a or not b:
            continue
        base_a = slug_a.split("/", 1)[-1]
        base_b = slug_b.split("/", 1)[-1]
        comp_slug = f"comparison/{base_a}-vs-{base_b}"
        user_msg = (
            f'<subject_a slug="{slug_a}" title="{a.title}">\n{a.content or ""}\n</subject_a>\n'
            f'<subject_b slug="{slug_b}" title="{b.title}">\n{b.content or ""}\n</subject_b>'
        )
        try:
            @wiki_llm_retry()
            async def _call():
                return await client.chat(
                    [
                        {"role": "system", "content": get_prompt("wiki", "WikiComparisonPrompt")},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.3,
                    max_tokens=4096,
                    purpose="wiki:comparison",
                )

            summary, content = _split_summary(await _call())
            if content:
                _upsert_page(session, kb_id, tenant_id, comp_slug, "comparison",
                             f"{a.title} vs {b.title}", content, summary)
        except Exception:
            continue
