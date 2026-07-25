from __future__ import annotations

from .base import ParseResult
from .registry import register


@register("md")
def parse_markdown(data: bytes, filename: str) -> ParseResult:
    return ParseResult(markdown=data.decode("utf-8", errors="replace"))


@register("txt")
def parse_text(data: bytes, filename: str) -> ParseResult:
    return ParseResult(markdown=data.decode("utf-8", errors="replace"))


@register("markdown")
def parse_markdown_alt(data: bytes, filename: str) -> ParseResult:
    return parse_markdown(data, filename)
