from __future__ import annotations

import re

from .hybrid import Candidate

_WS = re.compile(r"\s+")
OVERLAP_THRESHOLD = 0.85  # WeKnora removePartialOverlaps 同款
_SKIP_TYPES = ("faq", "wiki_page")


def _normalize(text: str) -> str:
    return _WS.sub("", (text or "").lower())


def _tokens(text: str) -> set[str]:
    return set(_WS.sub(" ", (text or "").lower()).split())


def dedup_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """签名去重 + 包含/token 重叠去重（低分者出局）；faq/wiki_page 语义完整不参与。"""
    kept: list[Candidate] = []
    seen_sig: set[str] = set()
    for c in candidates:
        if c.chunk_type in _SKIP_TYPES:
            kept.append(c)
            continue
        norm = _normalize(c.content)
        if norm and norm in seen_sig:
            continue
        if norm:
            seen_sig.add(norm)
        kept.append(c)

    removed: set[int] = set()
    for i in range(len(kept)):
        if i in removed or kept[i].chunk_type in _SKIP_TYPES:
            continue
        ni = _normalize(kept[i].content)
        if not ni:
            continue
        for j in range(i + 1, len(kept)):
            if j in removed or kept[j].chunk_type in _SKIP_TYPES:
                continue
            nj = _normalize(kept[j].content)
            if not nj:
                continue
            short_idx, long_idx = (i, j) if len(ni) <= len(nj) else (j, i)
            short_n, long_n = (ni, nj) if short_idx == i else (nj, ni)
            if short_n not in long_n:
                ti, tj = _tokens(kept[i].content), _tokens(kept[j].content)
                smaller = ti if len(ti) <= len(tj) else tj
                if not smaller or len(ti & tj) / len(smaller) < OVERLAP_THRESHOLD:
                    continue
            victim = short_idx if kept[short_idx].score <= kept[long_idx].score else long_idx
            removed.add(victim)
    return [c for k, c in enumerate(kept) if k not in removed]
