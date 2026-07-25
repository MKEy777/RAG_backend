from __future__ import annotations

from ..settings import settings
from .heading import ChunkSpan, split_by_heading
from .heuristic import split_heuristic
from .parent_child import ParentChildPair, build_parent_child
from .profiler import analyze, choose_strategy
from .recursive import split_recursive
from .validator import validate


def chunk_document(
    text: str,
    strategy: str = "auto",
    chunk_size: int | None = None,
    overlap: int | None = None,
    enable_parent_child: bool = True,
    parent_size: int | None = None,
    child_size: int | None = None,
) -> list[ParentChildPair] | list[ChunkSpan]:
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    if strategy == "auto":
        profile = analyze(text)
        strategy = choose_strategy(profile)

    chunks = _run_strategy(text, strategy, chunk_size, overlap)

    if not validate(chunks, chunk_size, len(text)):
        fallback_order = ["heading", "heuristic", "recursive"]
        for fallback in fallback_order:
            if fallback == strategy:
                continue
            candidate = _run_strategy(text, fallback, chunk_size, overlap)
            if validate(candidate, chunk_size, len(text)):
                chunks = candidate
                break
        else:
            chunks = split_recursive(text, chunk_size, overlap)

    if enable_parent_child:
        return build_parent_child(chunks, parent_size, child_size)
    return chunks


def _run_strategy(
    text: str, strategy: str, chunk_size: int, overlap: int
) -> list[ChunkSpan]:
    if strategy == "heading":
        return split_by_heading(text, chunk_size)
    if strategy == "heuristic":
        return split_heuristic(text, chunk_size, overlap)
    return split_recursive(text, chunk_size, overlap)
