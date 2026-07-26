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


def split_by_heading(
    text: str, chunk_size: int = 512, whole_sections: bool = False
) -> list[ChunkSpan]:
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

        raw = text[seg_start:seg_end]
        seg_text = raw.strip()
        # start/end 精确对齐 stripped content，保证 end - start == len(text)
        lstrip_n = len(raw) - len(raw.lstrip())
        actual_start = seg_start + lstrip_n
        title_path = [h[1] for h in heading_stack]
        sections.append(ChunkSpan(text=seg_text, title_path=title_path, start=actual_start, end=actual_start + len(seg_text)))

    if matches[0].start() > 0:
        raw_pre = text[: matches[0].start()]
        preamble = raw_pre.strip()
        if preamble:
            lstrip_n = len(raw_pre) - len(raw_pre.lstrip())
            sections.insert(0, ChunkSpan(text=preamble, start=lstrip_n, end=lstrip_n + len(preamble)))

    # 父子模式：父块 = 标题层级完整 section，不再细分
    if whole_sections:
        return sections

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


_BREAK_CHARS = set("。？！；\n.?!;")
_FORMULA_START_RE = re.compile(
    r"^[\U0001D400-\U0001D7FF\u2211\u220F\u222B\u2202\u2207\u221A]"
    r"|^[a-zA-Z_][a-zA-Z0-9_]*\s*[=≈]"
    r"|^\$\$"
)


def _find_break_point(text: str, target: int, window: int = 100) -> int:
    """Find a natural break point near target position."""
    if target >= len(text):
        return len(text)
    lo = max(1, target - window)
    hi = min(len(text), target + window)

    # Prefer paragraph boundary
    best = text.rfind("\n\n", lo, hi)
    if best >= lo:
        return best + 2

    # Prefer newline after sentence-ending punct (keeps formulas with intro)
    for i in range(target, lo, -1):
        if text[i - 1] == "\n" and i >= 2 and text[i - 2] in "。？！；.?!;：:":
            return i

    # Single newline before target
    best = text.rfind("\n", lo, target)
    if best >= lo:
        return best + 1

    # Sentence-ending punctuation
    for i in range(target, lo, -1):
        if text[i - 1] in _BREAK_CHARS:
            return i

    return target


def _greedy_split(text: str, chunk_size: int) -> list[str]:
    segments = _split_respecting_fences(text)
    chunks: list[str] = []
    current = ""
    for seg in segments:
        if seg.startswith("\x00FENCE"):
            if current.strip():
                chunks.append(current.strip())
                current = ""
            chunks.append(seg[6:])
        else:
            paragraphs = seg.split("\n\n")
            for para in paragraphs:
                if not para.strip():
                    continue
                if len(para) > chunk_size:
                    if current.strip():
                        chunks.append(current.strip())
                        current = ""
                    pos = 0
                    while pos < len(para):
                        end = pos + chunk_size
                        if end < len(para):
                            end = _find_break_point(para, end)
                        chunks.append(para[pos:end].strip())
                        pos = end
                elif len(current) + len(para) + 2 > chunk_size and current:
                    chunks.append(current.strip())
                    current = para
                else:
                    current = current + "\n\n" + para if current else para
    if current.strip():
        chunks.append(current.strip())
    # Merge formula-only chunks back into previous
    merged: list[str] = []
    for c in chunks:
        if merged and _FORMULA_START_RE.match(c) and len(merged[-1]) + len(c) + 1 <= chunk_size * 2:
            merged[-1] = merged[-1] + "\n" + c
        else:
            merged.append(c)
    return merged if merged else [text]


def _split_respecting_fences(text: str) -> list[str]:
    """Split text into segments; code fences are returned as atomic units prefixed with marker."""
    segments: list[str] = []
    lines = text.split("\n")
    current_lines: list[str] = []
    in_fence = False
    fence_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not in_fence and stripped.startswith("```"):
            if current_lines:
                segments.append("\n".join(current_lines))
                current_lines = []
            in_fence = True
            fence_lines = [line]
        elif in_fence and stripped.startswith("```"):
            fence_lines.append(line)
            segments.append("\x00FENCE" + "\n".join(fence_lines))
            fence_lines = []
            in_fence = False
        elif in_fence:
            fence_lines.append(line)
        else:
            current_lines.append(line)
    if in_fence and fence_lines:
        segments.append("\x00FENCE" + "\n".join(fence_lines))
    if current_lines:
        segments.append("\n".join(current_lines))
    return segments


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
