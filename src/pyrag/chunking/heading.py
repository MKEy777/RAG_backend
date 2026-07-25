from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class ChunkSpan:
    text: str
    title_path: list[str] = field(default_factory=list)
    start: int = 0
    end: int = 0


def split_by_heading(text: str, chunk_size: int = 512) -> list[ChunkSpan]:
    matches = [m for m in _HEADING_RE.finditer(text) if not _inside_code_fence(text, m.start())]
    if not matches:
        return [ChunkSpan(text=text, start=0, end=len(text))]

    sections: list[ChunkSpan] = []
    heading_stack: list[tuple[int, str]] = []

    for i, m in enumerate(matches):
        seg_start = m.start()
        seg_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        level = len(m.group(1))
        title = m.group(2).strip()

        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))

        seg_text = text[seg_start:seg_end].strip()
        title_path = [h[1] for h in heading_stack]
        sections.append(ChunkSpan(text=seg_text, title_path=title_path, start=seg_start, end=seg_end))

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.insert(0, ChunkSpan(text=preamble, start=0, end=matches[0].start()))

    result: list[ChunkSpan] = []
    for sec in sections:
        if len(sec.text) <= chunk_size * 2:
            result.append(sec)
        else:
            sub_chunks = _greedy_split(sec.text, chunk_size)
            for j, sub in enumerate(sub_chunks):
                offset = sec.start + sec.text.find(sub)
                result.append(ChunkSpan(
                    text=sub,
                    title_path=sec.title_path,
                    start=offset,
                    end=offset + len(sub),
                ))
    return result


def _greedy_split(text: str, chunk_size: int) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(para), chunk_size):
                chunks.append(para[i : i + chunk_size])
        elif len(current) + len(para) + 2 > chunk_size and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text]


def _inside_code_fence(text: str, pos: int) -> bool:
    """Check if position is inside a ``` code fence."""
    fence_count = 0
    idx = 0
    while idx < pos:
        nl = text.find("\n", idx)
        if nl == -1 or nl >= pos:
            break
        line = text[idx:nl].strip()
        if line.startswith("```"):
            fence_count += 1
        idx = nl + 1
    return fence_count % 2 == 1
