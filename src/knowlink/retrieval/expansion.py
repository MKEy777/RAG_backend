from __future__ import annotations

import re

_QUESTION_WORDS = [
    "什么是", "什么", "如何", "怎么样", "怎么", "怎样", "为什么", "为何",
    "哪些", "哪个", "是否", "多少", "请问",
]
_DELIMS = re.compile(r"[，,。;；:：?？!！]")
_QUOTED = re.compile(r"[\"""'『「](.+?)[\"""'』」]")
MAX_EXPANSIONS = 5  # WeKnora query_expansion 同款上限
_MIN_LEN = 4


def expand_queries(query: str) -> list[str]:
    query = (query or "").strip()
    if len(query) < _MIN_LEN:
        return []
    out: list[str] = []
    seen = {query.lower()}

    def add(s: str) -> None:
        s = re.sub(r"\s+", " ", s).strip(" ？?！!。，,")
        if len(s) >= _MIN_LEN and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)

    cleaned = query
    for w in _QUESTION_WORDS:
        cleaned = cleaned.replace(w, " ")
    add(cleaned)

    for seg in _DELIMS.split(query):
        if len(seg.strip()) > 5:
            add(seg)

    for m in _QUOTED.finditer(query):
        add(m.group(1))

    return out[:MAX_EXPANSIONS]
