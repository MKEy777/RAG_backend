from __future__ import annotations

import struct
from pathlib import Path
from typing import Callable

from ..logging import get_logger
from .base import ParseResult

OLE_MAGIC = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"
PDF_MAGIC = b"%PDF"
ZIP_MAGIC = b"PK\x03\x04"

_PARSERS: dict[str, Callable[[bytes, str], ParseResult]] = {}

log = get_logger("parsing")


def register(ext: str):
    def decorator(fn: Callable[[bytes, str], ParseResult]):
        _PARSERS[ext.lower()] = fn
        return fn

    return decorator


def detect_format(data: bytes, filename: str) -> str:
    if data[:8] == OLE_MAGIC:
        ext = Path(filename).suffix.lower()
        if ext == ".docx":
            return "doc"
        if ext == ".pptx":
            return "ppt"
        if ext in (".xlsx", ".xls"):
            return "xls"
        return "ole"
    if data[:4] == PDF_MAGIC:
        return "pdf"
    if data[:4] == ZIP_MAGIC:
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext in ("docx", "pptx", "xlsx"):
            return ext
        return "zip"
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext if ext else "txt"


def parse_file(data: bytes, filename: str, base_dir: str | None = None) -> ParseResult:
    fmt = detect_format(data, filename)
    log.info("文件格式检测", filename=filename, format=fmt, size=len(data))
    parser = _PARSERS.get(fmt)
    if parser is None:
        log.warning("未找到对应解析器_使用纯文本回退", filename=filename, format=fmt)
        parser = _PARSERS.get("txt")
    if parser is None:
        return ParseResult(markdown=data.decode("utf-8", errors="replace"))
    result = parser(data, filename, base_dir=base_dir)
    log.info("解析完成", filename=filename, format=fmt, markdown_len=len(result.markdown))
    return result
