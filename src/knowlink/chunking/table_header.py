from __future__ import annotations

import re

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_SEP_ROW = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")


def _col_count(line: str) -> int:
    return len(line.strip().strip("|").split("|"))


def _trailing_table_header(chunk: str) -> tuple[str, int] | None:
    """块尾仍处于表格中时，返回 (表头行+分隔行, 列数)。"""
    lines = chunk.split("\n")
    if not lines or not _TABLE_ROW.match(lines[-1]):
        return None
    header_idx = None
    for i in range(1, len(lines)):
        if _SEP_ROW.match(lines[i]) and _TABLE_ROW.match(lines[i - 1]):
            header_idx = i - 1
    if header_idx is None:
        return None
    return f"{lines[header_idx]}\n{lines[header_idx + 1]}", _col_count(lines[header_idx])


def _leading_row_cols(chunk: str) -> int | None:
    lines = chunk.split("\n")
    if lines and _TABLE_ROW.match(lines[0]) and not _SEP_ROW.match(lines[0]):
        return _col_count(lines[0])
    return None


def annotate_table_headers(chunks: list[str]) -> list[str]:
    """续块以表格行开头且上一块表格未闭合时前插表头（列数一致才注入）。"""
    if len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    active = _trailing_table_header(chunks[0])
    for chunk in chunks[1:]:
        cols = _leading_row_cols(chunk)
        if active and cols is not None and cols == active[1] and not chunk.startswith(active[0]):
            out.append(f"{active[0]}\n{chunk}")
        else:
            out.append(chunk)
        active = _trailing_table_header(chunk)
    return out
