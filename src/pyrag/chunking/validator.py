from __future__ import annotations

from .heading import ChunkSpan

MIN_CHUNK_CHARS = 50


def validate(chunks: list[ChunkSpan], chunk_size: int, total_text_len: int) -> bool:
    if not chunks:
        return False
    if total_text_len > chunk_size * 2 and len(chunks) == 1:
        return False
    small_count = sum(1 for c in chunks if len(c.text) < MIN_CHUNK_CHARS)
    if small_count > len(chunks) // 4 and small_count > 2:
        return False
    max_len = max(len(c.text) for c in chunks)
    if max_len < chunk_size // 4 and total_text_len > chunk_size:
        return False
    if max_len > chunk_size * 2:
        return False
    return True
