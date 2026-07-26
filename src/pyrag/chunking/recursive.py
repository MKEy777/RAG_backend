from __future__ import annotations

import re
import uuid

from ..settings import settings
from .heading import ChunkSpan

SEPARATORS = ["\n\n", "\n", "。", ".", "？", "?", "！", "!", "；", ";", "，", ","]

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_LATEX_BLOCK_RE = re.compile(r"\$\$[\s\S]*?\$\$")
_TABLE_BLOCK_RE = re.compile(r"(?:^\|.+\|[ ]*$\n?)+", re.MULTILINE)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]+\)")

PROTECT_PATTERNS = [_CODE_BLOCK_RE, _LATEX_BLOCK_RE, _TABLE_BLOCK_RE, _IMAGE_RE, _LINK_RE]


def _protect(text: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}
    for pattern in PROTECT_PATTERNS:
        def replacer(m: re.Match) -> str:
            matched = m.group(0)
            if len(matched) > settings.protect_block_limit:
                return matched
            key = f"\x00PH{uuid.uuid4().hex[:12]}\x00"
            placeholders[key] = matched
            return key

        text = pattern.sub(replacer, text)
    return text, placeholders


def _restore(text: str, placeholders: dict[str, str]) -> str:
    for key, val in placeholders.items():
        text = text.replace(key, val)
    return text


def split_recursive(
    text: str,
    chunk_size: int = 512,
    overlap: int = 80,
    separators: list[str] | None = None,
) -> list[ChunkSpan]:
    separators = separators or SEPARATORS
    protected, placeholders = _protect(text)
    pieces = _recursive_split(protected, separators, chunk_size)
    result: list[ChunkSpan] = []
    search_from = 0
    for piece in pieces:
        restored = _restore(piece, placeholders)
        stripped = restored.strip()
        if not stripped:
            continue
        # 在原文中定位实际起始位置，跳过被切分消耗的分隔符
        pos = text.find(stripped, search_from)
        if pos == -1:
            pos = search_from
        result.append(ChunkSpan(text=stripped, start=pos, end=pos + len(stripped)))
        search_from = pos + len(stripped)
    return result


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = separators[0] if separators else ""
    remaining_seps = separators[1:] if len(separators) > 1 else []

    if sep:
        parts = text.split(sep)
    else:
        parts = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current)
            if len(part) > chunk_size and remaining_seps:
                sub_chunks = _recursive_split(part, remaining_seps, chunk_size)
                chunks.extend(sub_chunks)
                current = ""
            elif len(part) > chunk_size:
                for i in range(0, len(part), chunk_size):
                    chunks.append(part[i : i + chunk_size])
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current)
    return chunks
