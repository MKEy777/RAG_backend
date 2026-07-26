from __future__ import annotations

from .base import ParseResult, ParserCapabilityError
from .registry import markitdown_convert, register


def _md_parser(fmt: str, suffix: str):
    def _parse(data: bytes, filename: str, **kwargs) -> ParseResult:
        text = markitdown_convert(data, suffix)
        if text is None:
            raise ParserCapabilityError(fmt, "markitdown")
        return ParseResult(markdown=text)

    return _parse


register("html")(_md_parser("html", ".html"))
register("htm")(_md_parser("htm", ".html"))
register("epub")(_md_parser("epub", ".epub"))
register("mhtml")(_md_parser("mhtml", ".mhtml"))
register("csv")(_md_parser("csv", ".csv"))
