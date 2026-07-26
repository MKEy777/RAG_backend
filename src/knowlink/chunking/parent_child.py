from __future__ import annotations

from dataclasses import dataclass, field

from ..settings import settings
from .heading import ChunkSpan
from .recursive import split_recursive


@dataclass
class ParentChildPair:
    parent_text: str
    parent_title_path: list[str]
    parent_start: int
    parent_end: int
    children: list[ChunkSpan] = field(default_factory=list)


def build_parent_child(
    chunks: list[ChunkSpan],
    parent_size: int | None = None,
    child_size: int | None = None,
) -> list[ParentChildPair]:
    child_size = child_size or settings.child_chunk_size
    child_overlap = child_size // 5

    pairs: list[ParentChildPair] = []
    for span in chunks:
        children = split_recursive(
            span.text, chunk_size=child_size, overlap=child_overlap
        )
        if not children:
            children = [ChunkSpan(text=span.text[:child_size], start=span.start, end=span.start + child_size)]

        pairs.append(ParentChildPair(
            parent_text=span.text,
            parent_title_path=span.title_path,
            parent_start=span.start,
            parent_end=span.end,
            children=children,
        ))
    return pairs
