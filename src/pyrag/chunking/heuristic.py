from __future__ import annotations

import re

from .heading import ChunkSpan

_FORM_FEED = "\f"
_CHAPTER_RE = re.compile(
    r"^(第[一二三四五六七八九十百千\d]+[章节篇部]|[A-Z][a-z]+\s+\d+|Chapter\s+\d+|Section\s+[\d.]+|\d+\.\d+\s+\S)",
    re.IGNORECASE,
)
_UPPER_TITLE_RE = re.compile(r"^[A-Z][A-Z\s]{3,}$")
_SEPARATOR_RE = re.compile(r"^[-*_]{3,}\s*$")
_PAGE_NUM_RE = re.compile(r"^(\d{1,3}|—\s*\d+\s*—)\s*$")


def split_heuristic(text: str, chunk_size: int = 512, overlap: int = 80) -> list[ChunkSpan]:
    boundaries = _detect_boundaries(text)
    segments = _split_at_boundaries(text, boundaries)
    chunks = _greedy_pack(segments, chunk_size, overlap)
    return chunks


def _detect_boundaries(text: str) -> list[int]:
    lines = text.split("\n")
    boundaries: list[int] = []
    offset = 0
    blank_run = 0

    for line in lines:
        stripped = line.strip()
        if _FORM_FEED in line:
            boundaries.append(offset)
        elif _CHAPTER_RE.match(stripped):
            boundaries.append(offset)
        elif _UPPER_TITLE_RE.match(stripped) and len(stripped) < 80:
            boundaries.append(offset)
        elif _SEPARATOR_RE.match(stripped):
            boundaries.append(offset)
        elif _PAGE_NUM_RE.match(stripped):
            boundaries.append(offset)
        elif not stripped:
            blank_run += 1
            if blank_run > 2:
                boundaries.append(offset)
        else:
            blank_run = 0
        offset += len(line) + 1

    return sorted(set(boundaries))


def _split_at_boundaries(text: str, boundaries: list[int]) -> list[str]:
    if not boundaries:
        return [text]
    segments: list[str] = []
    prev = 0
    for b in boundaries:
        if b > prev:
            seg = text[prev:b].strip()
            if seg:
                segments.append(seg)
        prev = b
    tail = text[prev:].strip()
    if tail:
        segments.append(tail)
    return segments if segments else [text]


def _greedy_pack(segments: list[str], chunk_size: int, overlap: int) -> list[ChunkSpan]:
    chunks: list[ChunkSpan] = []
    current = ""
    start = 0

    for seg in segments:
        if len(current) + len(seg) + 1 > chunk_size and current:
            chunks.append(ChunkSpan(text=current.strip(), start=start, end=start + len(current)))
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + "\n" + seg
            else:
                current = seg
            start = start + len(current) - len(seg)
        else:
            current = current + "\n" + seg if current else seg

    if current.strip():
        chunks.append(ChunkSpan(text=current.strip(), start=start, end=start + len(current)))
    return chunks if chunks else [ChunkSpan(text=text if segments else "", start=0, end=0)]
